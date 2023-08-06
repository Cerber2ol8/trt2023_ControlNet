rm myProfile.nsys -rep
~/nsight-systems-2023.2.1/bin/nsys profile -o myProfile -f true trtexec --loadEngine=./trt_dir/unet.engine --verbose
