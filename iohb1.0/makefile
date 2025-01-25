# Diretório com os headers
IDIR = ./iohb1.0


# Compilador e flags
CC = gcc
CFLAGS = -Wno-pointer-to-int-cast -Wno-format-security -I$(IDIR)
LDFLAGS = -lm


# Alvos
all: boing


# Alvo principal: compila o programa
boing: boing.o iohb.o
   $(CC) boing.o iohb.o $(CFLAGS) $(LDFLAGS) -o boing


# Compila o arquivo boing.c
boing.o: boing.c $(IDIR)/iohb.h
   $(CC) $(CFLAGS) -c boing.c -o boing.o


# Compila a biblioteca iohb
iohb.o: $(IDIR)/iohb.c $(IDIR)/iohb.h
   $(CC) $(CFLAGS) -c $(IDIR)/iohb.c -o iohb.o


# Limpa os arquivos gerados
clean:
   rm -f *.o boing

