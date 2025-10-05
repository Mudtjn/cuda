NVCC := nvcc
# Pass the CUDA file to build via command line: make build FILE=path\to\file.cu

build:
	@if not defined FILE ( \
		echo Error: Please provide the CUDA FILE, e.g., make build FILE=dir\foo.cu & exit /b 1 \
	) else ( \
		for %%F in ($(FILE)) do ( \
			set "outdir=%%~dpF" & \
			set "base=%%~nF" & \
			echo Building %%~nF from $(FILE)... & \
			$(NVCC) -o %%~dpF%%~nF.exe $(FILE) & \
			echo Output: %%~dpF%%~nF.exe \
		) \
	)

clean:
	@if not defined DIR ( \
		echo Error: Please provide the DIR variable, e.g., make clean DIR=a\ & exit /b 1 \
	) else ( \
		del /q $(DIR)\*.exe $(DIR)\*.exp $(DIR)\*.lib & \
		echo Deleted all .exe, .exp, and .lib files from $(DIR) \
	)

.PHONY: build clean