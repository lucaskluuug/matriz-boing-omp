IDIR = ./iohb1.0

default: all

all: boing
	gcc boing.c iohb.o -Wno-pointer-to-int-cast -Wno-format-security -I$(IDIR) -fopenmp -o boing

boing:
	gcc -Wno-pointer-to-int-cast -Wno-format-security -I$(IDIR) -c $(IDIR)/iohb.c
