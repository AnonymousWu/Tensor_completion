import os

jobname = "ctf_all_strongscaling"
# outfile = "all.out"
# errorfile = "all.error"
CTF_PPN = 64
OMP_NUM_THREADS = 1
queue = "normal"

for nodes in [1]:
#for nodes in [1,2,4,8,16,32,64,128,256]:
    mpitask = nodes * CTF_PPN
    time = "08:00:00"
    mail = "xwu74@illinois.edu"
    mailtype = "all"

    text_file = open("script_all_ss_N%s.sh" % (nodes), "w")

    text_file.write("#!/bin/bash\n")
    text_file.write("#----------------------------------------------------\n\n\n")
    text_file.write("#SBATCH -J %s\n" % jobname)
    text_file.write("#SBATCH -o all.bench.ss.N%s.o%%j.out\n" % (nodes))
    text_file.write("#SBATCH -e all.bench.ss.N%s.o%%j.err\n" % (nodes))	
    text_file.write("#SBATCH -p %s\n" % queue)
    text_file.write("#SBATCH -N %s\n" % nodes)
    text_file.write("#SBATCH -n %s\n" % mpitask)
    text_file.write("#SBATCH -t %s\n" % time)
    text_file.write("#SBATCH --mail-user= %s\n" % mail)
    text_file.write("#SBATCH --mail-type= %s\n\n" % mailtype)
    # text_file.write("#SBATCH -A %s\n\n\n" % myproject)

    text_file.write("module list\n")
    text_file.write("pwd\n")
    text_file.write("date\n\n")
    
    text_file.write('export LD_LIBRARY_PATH="/opt/apps/intel18/python2/2.7.15/lib:/opt/apps/libfabric/1.7.0/lib:/opt/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/lib:/opt/intel/debugger_2018/libipt/intel64/lib:/opt/intel/debugger_2018/iga/lib:/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64/gcc4.7:/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.2.199/linux/ipp/lib/intel64:/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/lib/intel64:/opt/apps/gcc/6.3.0/lib64:/opt/apps/gcc/6.3.0/lib:/opt/apps/xsede/gsi-openssh-7.5p1b/lib64:/opt/apps/xsede/gsi-openssh-7.5p1b/lib:::/home1/06115/tg854143/ctf/lib_shared:/home1/06115/tg854143/ctf/lib_python:/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64"\n')
    text_file.write('export PYTHONPATH="/opt/apps/intel18/impi18_0/python2/2.7.15/lib/python2.7/site-packages:/home1/06115/tg854143/ctf/lib_python"\n\n')
    

    # parameters
    regParam = 0.1
    sgd_stepSize = 0.01
    sgd_sample_rate = 0.01
    use_func = 0
    num_iter = 1
    err_thresh = .001
    run_implicit = 1
    run_explicit = 1

    #for s in [2000,4000,8000]:
        #for sparsity in [.01,.001,.0001]:
            #for r in [10,40,160]:
                #for block_size in [s/2, s/4, s/8]:	
    for s in [100]:
        for sparsity in [.001]:
            for r in [10]:
                for block_size in [s]:
                    text_file.write("ibrun python ../Tensor_completion/test/combined_test.py %s %s %s %s %s %s %s %s %s %s %s %s %s %s \n" % (s,s,s,sparsity,r,regParam, sgd_stepSize, sgd_sample_rate,block_size,use_func,num_iter,err_thresh,run_implicit,run_explicit))
	
    text_file.close()
