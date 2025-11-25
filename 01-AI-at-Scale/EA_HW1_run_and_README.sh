#
#
# HW1 E. D. Araya
#
#
# Problem A
echo "****** Problem A: "
echo "-Instructions: Make sure that the single GPU code runs on Polaris."
echo "  Code provided in the lecture was copied to a .py script,"
echo "  edited and a .sh script was modified to run it"
echo "-Solution: The code successfully runs" 
#bash EA_HW1_A_run_pytorch_single_GPU_example.sh

# Problem B
echo "****** Problem B:" 
echo "-Instructions: The counting of ranks, does not necessarily has to be"
echo "  a mix-and-match between mpi4py and PALS. Try to implement the rank"
echo "  counting method using just PALS or mpi4py.device_count() methods can be useful here."
echo "-Solution: Checking online, there seems to be no mpi4py.device_count(), thus" 
echo "  the hint given in the instructions was unclear. torch.cuda.device_count() was"
echo "  used instead."
#bash EA_HW1_B_run_ddp_pytorch_2p8_N1_R4.sh

# Problem C
echo "***** Problem C:"
echo "-Instructions: Play with different dimensions of the src and tgt tensors."
echo "-Solution: the script pytorch_2p8_ddp.py was modified as follows: "
echo "  src:(2048, 1, 512), tgt:(2048, 20, 512) = total train time: 4.62s"
echo "  src:(4096, 1, 512), tgt:(4096, 20, 512) = total train time: 9.01s"
echo "  Computation time increased slightly slower than linear. "
echo "   **** src:(2048, 1, 512), tgt:(2048, 20, 512) ****"
#bash run_ddp_pytorch_2p8_N1_R4.sh
echo "   **** src:(4096, 1, 512), tgt:(4096, 20, 512) **** "
#bash EA_HW1_C_run_ddp_pytorch_2p8_N1_R4_TEST1.sh
echo "   **** Several tests were tried with higher dimensions, e.g.,"
echo "   src:(2048,2048, 1, 512), tgt:(2048,2048, 20, 512) that resulted in errors. "
echo "   It was unclear how to fix them."
#bash EA_HW1_C_run_ddp_pytorch_2p8_N1_R4_TEST2.sh

# Problem D
echo "***** Problem D:"
echo "-Instructions: Explore the cost of collective communication, by setting up"
echo "   a scenario, where you have only two ranks, but each rank resides on a "
echo "   different node. Profile and try to reason about the results." 
echo "-Solution: A copy of the script run_ddp_prof_pytorch_2p8_N1_R4.sh was"
echo "   created, then NODES was changed to 2. The script ran, and resulted in"
echo "   x3203c0s13b1n0.hsn.cm.polaris.alcf.anl.gov 0: total train time: 10.81s"
echo "   The original run_ddp_prof_pytorch_2p8_N1_R4.sh resulted in total train time: 12.12s,"
echo "   so it seems the NODES=2 improves efficiency, although as expected, collective "
echo "   communication would make the improvements significantly less than linear."
#bash EA_HW1_D_run_ddp_prof_pytorch_2p8_N1_R4.sh

# Problem E
echo "***** Problem E:"
echo "-Instructions: Try other file formats to explore the I/O bottleneck."
echo "-Solution: Original pytorch_2p8_ddp_hdf5_prof.py results in: total train time: 10.83s"
echo "   The file pytorch_2p8_ddp_hdf5_prof.py was modified in EA_HW1_E_pytorch_2p8_ddp_hdf5_prof.py"
echo "   to instead of saving the tensor and reading it back in .h5 format, the tensor was saved"
echo "   and read back in .npz format. There was not much change in the total train time: 10.73s"
echo "   An example of this type of exercise would have been useful during the lecture to better"
echo "   achieve the goal of this part of the homework."
#bash run_ddp_prof_pytorch_2p8_hdf5_N1_R4.sh
#bash EA_HW1_E_run_ddp_prof_pytorch_2p8_hdf5_N1_R4.sh

# Problem F
echo "***** Problem F:"
echo "-Instructions: Make the tensors really large, specially the 2nd and 3rd dimension and explore different data types."
echo "-Solution: A test with run_ddp_prof_pytorch_2p8_hdf5_compile_N1_R4.sh resulted in: total train time: 11.42s"
echo "   The .py file of the example was copied to EA_HW1_F_pytorch_2p8_ddp_hdf5_compile_prof.py, and the tensors"
echo "   were changed to src = torch.rand((10*2048, 1, 512)), tgt = torch.rand((10*2048, 20, 512)), resulting in"
echo "   total train time: 104.65s, which shows a computation time that increased approximately linearly with"
echo "   the size of the dataset."
echo "   When the tensors were modified as follows: src = torch.rand((1*2048, 1, 4*512)) "
echo "   tgt = torch.rand((1*2048, 20,4*512)), the test resulted in the following error:"
echo "           RuntimeError: the feature number of src and tgt must be equal to d_model"
echo "   It was unclear how to correct the error."
bash EA_HW1_F_run_ddp_prof_pytorch_2p8_hdf5_compile_N1_R4.sh
