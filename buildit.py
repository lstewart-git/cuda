
import os


class ExeBuilder:
    
    def __init__(self):
        self.cu_compiler = 'nvcc'

        self.cu_compiler_flags = '-Xptxas=-v -O3 -m64 '\
                                '--ftz=true -prec-div=false -prec-sqrt=false '\
                                '-I/usr/local/cuda/include -I/usr/local/cuda/src'\
                                '-DCUBLAS -DCUBLAS_GFORTRAN'

        # set architecture here

        self.cu_compiler_flags = self.cu_compiler_flags + '  -gencode arch=compute_60,code=sm_60'

                    #CUDA 8 FLAGS
        self.cu_libraries = '-I/usr/local/magma/include -I/usr/local/cuda/include '\
                        '-L/usr/local/magma/lib -lmagma '\
                        '-L/usr/local/cuda/lib64 -lcublas -lcusparse -lcudart -cxxlib'

        self.libraries = self.libraries + ' ' + self.cu_libraries


    def run_me(self):
        file_name = 'myk.cu'
        compiler_flags = self.cu_compiler_flags
        os.system(self.cu_compiler + ' -c ' + compiler_flags + ' ' + file_name)

# MAIN DRIVER STARTS HERE    
if __name__ == "__main__":

    run_obj = ExeBuilder()
    run_obj.run_me()


