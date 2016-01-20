import os
import ROOT

#os.environ['PYOPENCL_CTX']='0' ##set value to select device for opencl compilation
#os.environ['CUDA_DEVICE'] ='0' ##set value to select device for cuda compilation

from time import time

import numpy as np
try:
    import pyopencl as opencl
except ImportError:
    opencl = None
    print "ImportError: pyopencl module was not found"

try:
    import pycuda as cuda
    import pycuda.autoinit
    import pycuda.driver as drv
    from   pycuda.compiler import SourceModule
except ImportError:
    cuda = None
    print "ImportError: pycuda module was not found"

a_float_t = np.float32 ## np.float64
a_int_t   = np.int32

class kernel:
    ''' class to handle much of the boiler plate code for evaluating ME kernel '''
    def __init__(self, nexternal, ndim, parobj, pdfsetn='CT10', kernelfn = 'kernel/kernel.cl', mode='opencl', comp=True, pR = None):
        self.mode = mode.lower()
        self.kernelfn = kernelfn
        self.pdfsetn = pdfsetn
        self.nexternal = nexternal
        self.ndim = ndim
        self.pars = np.ndarray(shape=(parobj.get_num_external_parameters(),), dtype=a_float_t, buffer=parobj.get_external_parameters())
        self.xi = None
        self.xp = None
        self.y = None
        self.rand = None
        self._load_pdf(pdfsetn)
        self._init_gpu(mode, comp)
        self.pR = pR

    def _load_pdf(self, pdfsetn):
        pdf_obj = ROOT.PDFGrid.load('CT10')
        self.pdf_data = np.ndarray(shape=(pdf_obj.get_num_grids()*pdf_obj.get_nx()*pdf_obj.get_nQ(),), dtype=a_float_t,
                                   buffer=pdf_obj.get_grid_data())
        self.pdf_bnds = np.array([pdf_obj.get_xmin(), pdf_obj.get_xmax(), pdf_obj.get_Qmin(), pdf_obj.get_Qmax()], dtype=a_float_t)

    def _init_gpu(self, mode, comp):
        if mode.lower() == 'opencl':
            memopts = { 'RW':opencl.mem_flags.READ_WRITE,
                        'RO':opencl.mem_flags.READ_ONLY,
                        'W' :opencl.mem_flags.WRITE_ONLY }
            self.ctx = opencl.create_some_context()
            self.ctx_name = str(self.ctx).split("'")[1]
            self.queue = opencl.CommandQueue(self.ctx)
            if comp:
                self.module = opencl.Program(self.ctx, open(self.kernelfn).read()).build(options=['-x clc++', '-D_CL_CUDA_READY_ -D_OPENCL_ -I%(cwd)s/..  -I%(cwd)s/src -I%(cwd)s/kernel' % {'cwd':os.getcwd()}])
            for vardef in ['pdf_data/RO', 'pdf_bnds/RO', 'pars/RO']:
                var, mem_t = vardef.split('/')
                setattr(self, var+'_device', opencl.Buffer(self.ctx, memopts[mem_t], size=getattr(self,var).nbytes))
                opencl.enqueue_copy(self.queue, getattr(self,var+'_device'), getattr(self,var))
        elif mode == 'cuda':
            self.ctx_name = pycuda.autoinit.device.name()
            mod = SourceModule("#define _CL_CUDA_READY_\n#define _CUDA_" + open(self.kernelfn).read(),include_dirs=['%(cwd)s/src'% {'cwd':os.getcwd()}, '%(cwd)s/kernel' % {'cwd':os.getcwd()}],no_extern_c=True)
            self.eval_cuda=mod.get_function('eval')

    def start_timer(self):
        self.tstart = time()
        self.time = -1

    def stop_timer(self):
        self.time = time() - self.tstart

    def __call__(self, xp=np.array([],dtype=a_float_t)):
        npts = max(len(xp),len(self.xi)/40)
        self.xp = np.asarray(xp).flatten().astype(a_float_t)
        if len(self.xp) == 0:
            self.xp = self.xi.copy()
        self.y = np.zeros(npts, dtype=a_float_t)
        if self.mode == 'opencl':
            memopts = { 'RW':opencl.mem_flags.READ_WRITE,
                        'RO':opencl.mem_flags.READ_ONLY,
                        'W' :opencl.mem_flags.WRITE_ONLY }
            for vardef in ['xi/RO', 'xp/RO', 'y/RW']:
                var, mem_t = vardef.split('/')
                setattr(self, var+'_device', opencl.Buffer(self.ctx, memopts[mem_t], size=getattr(self,var).nbytes))
                opencl.enqueue_copy(self.queue, getattr(self,var+'_device'), getattr(self,var), is_blocking = False)

            self.module.eval(self.queue, self.y.shape, None, self.xi_device, self.xp_device, self.y_device, self.pdf_data_device, self.pdf_bnds_device, self.pars_device)

            opencl.enqueue_copy(self.queue, self.y, self.y_device, is_blocking = False)

            return self.y.astype(np.float64)

        elif self.mode == 'cuda':

            numDict = dict()
            for lab,key in zip([2**i for i in xrange(3,20)], [2**j for j in xrange(11,28)]): numDict[key] = lab
            numBlocks =numDict[self.pR]
            # TODO
            # Make it 2D into the kernel
            threadsPerBlock = len(self.y)/(numBlocks) #len(self.y) = len(self.xp)/4
            self.eval_cuda(drv.In(self.xi),drv.In(self.xp),drv.InOut(self.y),drv.In(self.pdf_data),drv.In(self.pdf_bnds), drv.In(self.pars),block=(threadsPerBlock,1,1),grid=(numBlocks,1,1))
            return self.y.astype(np.float64)

    def eval(self, xp=np.array([],dtype=a_float_t)):
        return self(xp)

    def set_momenta(self, xi):
        if len(xi) != 4*self.nexternal:
            print len(xi), 4*self.nexternal
            raise ValueError
        self.xi = xi.flatten()

