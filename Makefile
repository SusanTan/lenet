CC=clang
CFLAGS= -ggdb
#OBJ = mnist.o lenet.o lenet_test.o conv2d.o MaxPool2d.o Linear.o
OBJ = mnist.o lenet.o conv2d.o Linear.o lenet_test.o MaxPool2d.o
LINKFLAGS= -lm
INPUT= mnist_train.csv

%.o: %.c
		$(CC) -c -o $@ $< $(CFLAGS)

lenet: $(OBJ)
		$(CC) -o $@ $^ $(CFLAGS) $(LINKFLAGS)

run: lenet
	./lenet

clean:
	rm *.o lenet

