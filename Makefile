CC=clang
CFLAGS= -ggdb
OBJ = mnist.o lenet.o lenet_test.o conv2d.o MaxPool2d.o Linear.o
LINKFLAGS= -lm
INPUT= mnist_train.csv

%.o: %.c
		$(CC) -c -o $@ $< $(CFLAGS)

lenet: $(OBJ)
		$(CC) -o $@ $^ $(CFLAGS) $(LINKFLAGS)

clean:
	rm *.o lenet

run: lenet
	./lenet $(INPUT)
