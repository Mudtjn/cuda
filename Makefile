NVCC := nvcc

# Usage: make build FILE=path/to/file.cu

build:
ifeq ($(strip $(FILE)),)
	$(error Please provide the CUDA FILE, e.g., make build FILE=src/foo.cu)
endif
	@outdir := $(dir $(FILE))
	@base := $(basename $(notdir $(FILE)))
	@echo Building $(base) from $(FILE)...
	@$(NVCC) -o $(outdir)$(base).exe $(FILE)
	@echo Output: $(outdir)$(base).exe

clean:
ifeq ($(strip $(DIR)),)
	$(error Please provide the DIR variable, e.g., make clean DIR=build/)
endif
	@rm -f $(DIR)/*.exe $(DIR)/*.exp $(DIR)/*.lib
	@echo Deleted all .exe, .exp, and .lib files from $(DIR)

.PHONY: build clean
