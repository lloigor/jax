
::goto targetted

E:\Install\Bazel\bazel-7.4.1.exe ^
 --output_user_root=D:\BazelRoot^
 run^
 --repo_env=HERMETIC_PYTHON_VERSION=3.12^
 --verbose_failures=true^
 --enable_workspace=true^
 --override_repository=xla=D:\Programs\oss\xla^
 --override_repository=llvm-raw=D:\Programs\oss\llvm-project^
 --override_repository=triton=D:\Programs\oss\triton-xla^
 --override_repository=cutlass_archive=D:\Programs\oss\cutlass^
 --cpu=x64_windows^
 --config=mkl_open_source_only^
 --config=avx_windows^
 --config=cuda^
 --config=build_cuda_with_nvcc^
 --action_env CUDA_TOOLKIT_PATH="D:\Install\CUDA\v12.6"^
 --action_env TF_CUDA_VERSION=12.6^
 --action_env TF_CUBLAS_VERSION=12.6.4^
 --action_env CUDNN_INSTALL_PATH="D:\Install\CUDNN\v9.6"^
 --action_env TF_CUDNN_VERSION=9.6.0^
 --action_env TF_CUDA_PATHS="D:\Install\CUDA\v12.6,D:\Install\CUDNN\v9.6"^
 --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_86^
 //jaxlib/tools:build_wheel^
 --^
 --output_path=D:\Programs\oss\jax\dist^
 --cpu=AMD64^
 --jaxlib_git_hash=""
goto done

:targetted
E:\Install\Bazel\bazel-7.4.1.exe ^
 --output_user_root=D:\BazelRoot^
 build^
 --repo_env=HERMETIC_PYTHON_VERSION=3.12^
 --verbose_failures=true^
 --enable_workspace=true^
 --override_repository=xla=D:\Programs\oss\xla^
 --override_repository=llvm-raw=D:\Programs\oss\llvm-project^
 --override_repository=triton=D:\Programs\oss\triton-xla^
 --override_repository=cutlass_archive=D:\Programs\oss\cutlass^
 --cpu=x64_windows^
 --config=mkl_open_source_only^
 --config=avx_windows^
 --config=cuda^
 --config=build_cuda_with_nvcc^
 --action_env CUDA_TOOLKIT_PATH="D:\Install\CUDA\v12.6"^
 --action_env TF_CUDA_VERSION=12.6^
 --action_env TF_CUBLAS_VERSION=12.6.4^
 --action_env CUDNN_INSTALL_PATH="D:\Install\CUDNN\v9.6"^
 --action_env TF_CUDNN_VERSION=9.6.0^
 --action_env TF_CUDA_PATHS="D:\Install\CUDA\v12.6,D:\Install\CUDNN\v9.6"^
 --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_86^
 @onednn//:mkl_dnn

:done
