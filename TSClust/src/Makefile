EXT := cu 

OBJS := tsclust.o

#compiler/preprocessor options
INCS := -I/usr/local/cuda/include
PARAMS := -Xcompiler "-I/usr/share/R/include -fpic"
PARAMS13 := 

#linker options
LD_PARAMS := -Xlinker "-L/usr/lib64/R/lib -lR "
LIBS := -L/usr/local/cuda/lib64 -lcuda

TARGETS := tsclust.so

CC := /usr/local/cuda/bin/nvcc 

all: $(TARGETS) EXEC

$(TARGETS): $(OBJS)
	$(CC) -shared $(LD_PARAMS) $(LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(CC) -c $(INCS) $(PARAMS) $^ -o $@

EXEC : tsclust.cu
	$(CC) -g -arch=sm_11 tsclust.cu -o tsclust

clean:
	rm -rf *.o
	rm tsclust
	rm tsclust.so

.PHONY: all clean
