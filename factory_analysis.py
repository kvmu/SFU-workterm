import ROOT
import numpy as np
import root_numpy as rootnp
import kernel
import vegas as VEGAS
import os
import re
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from timer import timer

# Library used to lead the data in this script
ROOT.gSystem.Load('libExRootAnalysis.so') # Path not specified because Daniel has automatically taken care of this in "global" profile.

nexternal                 = 8 # Number of exernal lines
tlvdim                    = 4 # Number of dimensions in a 4-vector
ndim                      = nexternal*tlvdim # Number of dimensions in 1 truth event

###############################################################################
#                          Set script parameters                              #
###############################################################################
nevts = 1000
gpu                       = "cuda" # opencl or cuda
dof                       = 4 # Degrees of freedom (do not know incoming partons [able to calculate calculate] and neutrinos)
neval                     = 8192*2 # Number of function evaluations for integration
nitn                      = 5 # Number of interations of the VEGAS algoritm (integration)
# want_b_veto = False # Do you want to include b-veto
amtOfInfoPerEvent = 16 # The number of indicies in signal_event_data and kernel_eventData array per event ( 2l + 2j = 2*4 + 2*4 = 16)
intDomain = 1000
kernelList = ['vbf125', 'ttbar']


for kernelMEM in kernelList:

    basedir = os.getcwd()
    if kernelMEM != '':
        os.chdir(kernelMEM)

    ###############################################################################
    #                         Pull background and signal                          #
    ###############################################################################

    file_path = '/home/kvmu/vbf/vbf_ttbar_run1_tupleData'
    fileList = sorted(os.listdir(file_path))

    for file in fileList:

        process_name = file.split('.')[0]
        print "\n########## " + process_name + " is being analysed through "+ kernelMEM + " ME ##########"
        fh     = ROOT.TFile(file_path +'/'+ file, 'read') # Read the background file
        inTree     = fh.Get('Output') # Obtain the tree from file


        # Not using any cuts

        # if want_b_veto:
        #     cuts = '((nbt20==0))*(lepID0*lepID1<0)*((l0pt>22) || (l1pt>22))*(nj>=2)*(cjv_leadPt<20.)*(olv==1)*(mtt_TrackHWW_Clj < 66.1876)*(ne==1)*(nm==1)*(mll>10)'
        # else:
        #     cuts = '(lepID0*lepID1<0)*((l0pt>22) || (l1pt>22))*(nj>=2)*(cjv_leadPt<20.)*(olv==1)*(mtt_TrackHWW_Clj < 66.1876)*(ne==1)*(nm==1)*(mll>10)'

        # Create variable list to extract from tree and apply proper conversion from from (pt, eta, phi) coordinate system to (E, px, py, pz)
        varlist = [\
        # Energy for lepton 0 (puts only positive charge leptons here) / Assume mass negligible
        'sqrt( (lepID0 < 0)*(l0pt*cosh(lepEta0))**2 + (lepID1 < 0)*(l1pt*cosh(lepEta1))**2 )', \
        # P_x for lepton 0
        '(lepID0 < 0)*l0pt*cos(lepPhi0) + (lepID1 < 0)*l1pt*cos(lepPhi1) ',\
        # P_y for lepton 0
        '(lepID0 < 0)*l0pt*sin(lepPhi0)  + (lepID1 < 0)*l1pt*sin(lepPhi1)',\
        # P_z for lepton 0
        '(lepID0 < 0)*l0pt*sinh(lepEta0) + (lepID1 < 0)*l1pt*sinh(lepEta1)',\

        # Energy for lepton 1 (puts only negative charge leptons here) / Assume mass negligible
        'sqrt( (lepID0 > 0)*(l0pt*cosh(lepEta0))**2 + (lepID1 > 0)*(l1pt*cosh(lepEta1))**2 )', \
        # P_x for lepton 1
        '(lepID0 > 0)*l0pt*cos(lepPhi0) + (lepID1 > 0)*l1pt*cos(lepPhi1) ',\
        # P_y for lepton 1
        '(lepID0 > 0)*l0pt*sin(lepPhi0)  + (lepID1 > 0)*l1pt*sin(lepPhi1)',\
        # P_z for lepton 1
        '(lepID0 > 0)*l0pt*sinh(lepEta0) + (lepID1 > 0)*l1pt*sinh(lepEta1)',\

        # Energy for jet 0
        'jetE0',\
        # P_x for jet 0
        'jetPt0*cos(jetPhi0)',\
        # P_y for jet 0
        'jetPt0*sin(jetPhi0)',\
        # P_z for jet 0
        'jetPt0*sinh(jetEta0)',\

        # Energy for jet 1
        'jetE1',\
        # P_x for jet 1
        'jetPt1*cos(jetPhi1)',\
        # P_y for jet 1
        'jetPt1*sin(jetPhi1)',\
        # P_z for jet 1
        'jetPt1*sinh(jetEta1)',\

        # Event weight no bveto
        'w/MV120_85_EventWeight',\

        # Event weight with bveto
        'w',\
        ]

        # Pull data from tree
        dataRecord = rootnp.tree2rec(inTree, branches = varlist)#, selection = cuts)

        fh.Close() # Close background input file

        # Give names to data taken from dataRecord

        dataRecord.dtype.names = ('lepE0','lepPx0','lepPy0','lepPz0',\
                                  'lepE1','lepPx1','lepPy1','lepPz1',\
                                  'jetE0','jetPx0','jetPy0','jetPz0',\
                                  'jetE1','jetPx1','jetPy1','jetPz1',\
                                    'w_div_MV120_85_EventWeight','w',\
                                                                     )

        nevts = min(nevts, dataRecord.size)

        print "Number of "+ process_name +" Events: {0}".format(nevts)

        kernel_eventData = np.zeros((nevts, amtOfInfoPerEvent))

        # Populate the kernel_eventData matrix, use numpy vector assignment techniques (faster than for loop)
        kernel_eventData[:, 0*tlvdim + 0] = dataRecord.lepE0[0:nevts]
        kernel_eventData[:, 0*tlvdim + 1] = dataRecord.lepPx0[0:nevts]
        kernel_eventData[:, 0*tlvdim + 2] = dataRecord.lepPy0[0:nevts]
        kernel_eventData[:, 0*tlvdim + 3] = dataRecord.lepPz0[0:nevts]

        kernel_eventData[:, 1*tlvdim + 0] = dataRecord.lepE1[0:nevts]
        kernel_eventData[:, 1*tlvdim + 1] = dataRecord.lepPx1[0:nevts]
        kernel_eventData[:, 1*tlvdim + 2] = dataRecord.lepPy1[0:nevts]
        kernel_eventData[:, 1*tlvdim + 3] = dataRecord.lepPz1[0:nevts]

        kernel_eventData[:, 2*tlvdim + 0] = dataRecord.jetE0[0:nevts]
        kernel_eventData[:, 2*tlvdim + 1] = dataRecord.jetPx0[0:nevts]
        kernel_eventData[:, 2*tlvdim + 2] = dataRecord.jetPy0[0:nevts]
        kernel_eventData[:, 2*tlvdim + 3] = dataRecord.jetPz0[0:nevts]

        kernel_eventData[:, 3*tlvdim + 0] = dataRecord.jetE1[0:nevts]
        kernel_eventData[:, 3*tlvdim + 1] = dataRecord.jetPx1[0:nevts]
        kernel_eventData[:, 3*tlvdim + 2] = dataRecord.jetPy1[0:nevts]
        kernel_eventData[:, 3*tlvdim + 3] = dataRecord.jetPz1[0:nevts]

        kernel_eventData = kernel_eventData.flatten()

        print '########## Data Retrieval Complete ##########'
        print '.............................................'
        print '############# Initializing MEM ##############'


        #************************** User defined function ****************************#
        def setparams():
            # Read in the parameters of the model using SLHAReader
            card_reader = ROOT.SLHAReader("cards/params.dat")
            pobj = ROOT.parameters_sm()
            pobj.set_independent_parameters(card_reader)
            pobj.set_independent_couplings();
            pobj.set_dependent_parameters();
            pobj.set_dependent_couplings();
            return pobj
        #*****************************************************************************#

        ###############################################################################
        #                   Process Matrix Elements through kernel                    #
        ###############################################################################

        # Load shared library of kernel src/ files to ROOT
        ROOT.gROOT.SetBatch(0)
        ROOT.gSystem.Load('lib/libpdf.so')
        ROOT.gSystem.Load("lib/libme.so")

        # Set the parameters of the model using the parameter card(s)
        pobj = setparams()

        # Create kernel object
        MEObj = kernel.kernel(nexternal-dof, amtOfInfoPerEvent, pobj, mode = gpu, pdfsetn = 'CT10', kernelfn = "kernel/kernel.cl", pR = neval)

        # Initialize results arrays
        resultsVector = np.zeros(nevts, dtype = kernel.a_float_t)

        class batch_func(VEGAS.BatchIntegrand):
            def __init__(self, ME_kernel_object = None):
                self.ME_kernel_object = ME_kernel_object

            def __call__(self, neutrino_space):
                eval = self.ME_kernel_object.eval(xp = neutrino_space)
                return eval

        # Set the kinematic values for the each event in the kernel object in order to evaluate the ME.

        write_path = "/home/kvmu/vbf/vbf_plotting"
        kern_output = "/z_"+process_name+"data_"+kernelMEM+"mem.txt"
        kernOut = open(write_path + kern_output, 'w+')

        integral = VEGAS.Integrator([[-intDomain, intDomain]]*4)
        v = batch_func(ME_kernel_object = MEObj)

        timeList = []

        for ievt in range(nevts):
            iteration = 0
            Qfound = False
            MEObj.set_momenta((kernel_eventData[ievt*amtOfInfoPerEvent:(ievt+1)*amtOfInfoPerEvent]).astype(kernel.a_float_t))
            with timer() as localTime:
                while(iteration < 5):
                    integral(v, nitn = 3, neval = neval, nhcube_batch = neval) # Adapt the grid before saving results
                    resultant = integral(v, nitn = nitn, neval = neval, nhcube_batch = neval)
                    if resultant.Q > 0.2:
                        kernOut.write(str(resultant.mean) + "\t" + str(resultant.sdev) + "\t" + str(resultant.chi2) + "\t" + str(resultant.dof) + "\t" + str(resultant.Q) +  "\n")
                        Qfound = True
                        break
                    else:
                        iteration+=1
                if not Qfound:
                    kernOut.write(str(resultant.mean) + "\t" + str(resultant.sdev) + "\t" + str(resultant.chi2) + "\t" + str(resultant.dof) + "\t" + str(resultant.Q) +  "\n")
            timeList.append(localTime.secs)

        kernOut.close()

        totalRunTime = sum(timeList)
        timePerEvent = totalRunTime/nevts

        paramOutput = '/zz_run_parameters_'+process_name+'_'+kernelMEM+"mem.txt"
        paramOut = open(write_path+paramOutput, 'w+')


        paramOut.write('Number of events: \t %i \n' %(nevts))
        paramOut.write('Number of phase space points: \t %i \n' %neval)
        paramOut.write('Number of iterations: \t %i \n' %nitn)
        paramOut.write("Total run time: \t %.2f s" % totalRunTime)

        paramOut.close()

        print "Total run time: \t %.2f s" % totalRunTime
        print "Average time per event: \t %.2f s" % timePerEvent
        print 'Number of events: \t %i' %(nevts)
        print 'Number of phase space points: \t %i' %neval
        print 'Number of iterations: \t %i' %nitn

    os.chdir(basedir)
