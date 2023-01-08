.SUFFIXES: .cpp .o

CC = g++

# 修改为自己的文件
SRCS = ./comm/trt.cpp \
	   ./hrnet/hrnet.cpp \
       ./hrnet/test.cpp

OBJS = $(SRCS:.cpp=.o)

OUTPUT = test

# 先修改为自己的路径
CUDA_ROOT=/usr/local/cuda-11.3
TRT_ROOT=/opt/tools/TensorRT-8.2.3.0
OPENCV_ROOT=/opt/tools/opencv-3.4.16

CFLAGS = -I$(CUDA_ROOT)/include \
         -I$(TRT_ROOT)/include \
         -I$(OPENCV_ROOT)/include

LIBS += -L$(CUDA_ROOT)/lib64 -lcudart \
        -L$(TRT_ROOT)/lib -lnvinfer \
        -L$(OPENCV_ROOT)/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_dnn \
        -lpthread
               
all : $(OBJS)
	$(CC) -o $(OUTPUT) $(OBJS) $(LIBS)
	@echo "-- success --"

.cpp.o :
	$(CC) -O3 -std=c++11 -Wall $(CFLAGS) -o $@ -c $<

clean :
	-rm -f $(OBJS)
	-rm -f .core*
	-rm $(OUTPUT)
