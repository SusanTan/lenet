CC=clang
CFLAGS= -O3
OBJ = mnist.o lenet.o conv2d.o Linear.o lenet_test.o MaxPool2d.o
LINKFLAGS= -lm
INPUT= mnist_train.csv

%.o: %.c
		$(CC) -c -o $@ $< $(CFLAGS)

lenet: $(OBJ)
		$(CC) -o $@ $^ $(CFLAGS) $(LINKFLAGS)

run: lenet
	./lenet run 60000 10000

profile: lenet
	./lenet profile 6000 1000

python:
	python lenet.py

clean:
	rm -rf *.o lenet data

