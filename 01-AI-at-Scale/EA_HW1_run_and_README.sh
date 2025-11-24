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
bash EA_HW1_A_run_pytorch_single_GPU_example.sh

# Problem B
echo "****** Problem B:" 
echo "-Instructions: The counting of ranks, does not necessarily has to be"
echo "  a mix-and-match between mpi4py and PALS. Try to implement the rank"
echo "  counting method using just PALS or mpi4py.device_count() methods can be useful here."
echo "-Solution: Checking online, there seems to be no mpi4py.device_count(), thus" 
echo "  the hint given in the instructions was unclear. torch.cuda.device_count() was"
echo "  used instead."
bash EA_HW1_B_run_ddp_pytorch_2p8_N1_R4.sh




