// Server side C/C++ program to demonstrate Socket programming
#include "linux/connector.h"
#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <linux/netlink.h>
#include <stdlib.h>
#include <arpa/inet.h>

#define PORT 8080
#define SLOW_MSG_CNT 1

int  server_fd = -1;
FILE* out = NULL;

void check_usage(int argc, char** argv);

FILE* open_file(char* filename, char* spec);

void caught_signal(int sig);

void exit_program(int code);
void exit_program_err(int code, char* func);


int main(int argc, char** argv)
{
    int new_socket, valread;
    struct sockaddr_in address;
    struct cn_msg *cmsg;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[4096] = {0};
    unsigned short l, l2;
	int count = 0;
    int ret;

    /* Make sure usage is correct */
    check_usage(argc, argv);
    printf("waiting for fifo reader\n");
    /* Open and check log file */
    out = open_file(argv[1], "w");
    printf("found a fifo reader\n");
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
     
    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                                                  &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( PORT );
     
    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address, 
                                 sizeof(address))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, 
                       (socklen_t*)&addrlen))<0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    printf("established socket connection\n");  
    while(1)
    {       
        valread = recv(new_socket , buffer, sizeof(buffer),0);       
        cmsg = NLMSG_DATA(buffer);
		//if (count % SLOW_MSG_CNT == 0)
			//printf("received %d bytes: id: %d val: %d seq: %d clen: %d\n", cmsg->len, cmsg->id.idx, cmsg->id.val, cmsg->seq, cmsg->len);
          //  printf("data %s\n", cmsg->data);
		/* Log the data to file */
		l = (unsigned short) cmsg->len;
        l2 = htons(l);
		fwrite(&l2, 1, sizeof(unsigned short), out);
		ret = fwrite(cmsg->data, 1, l, out);
        //fflush (out);
        ++count;
    }
	exit_program(0);
	return 0;
}

void check_usage(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s <output_file>\n", argv[0]);
		exit_program(1);
	}
}

FILE* open_file(char* filename, char* spec)
{
	FILE* fp = fopen(filename, spec);
	if (!fp)
	{
		perror("fopen");
		exit_program(1);
	}
	return fp;
}

void caught_signal(int sig)
{
	fprintf(stderr, "Caught signal %d\n", sig);
	exit_program(0);
}

void exit_program(int code)
{
	if (out)
	{
		fclose(out);
		out = NULL;
	}
	if (server_fd != -1)
	{
		close(server_fd);
		server_fd = -1;
	}
	exit(code);
}

void exit_program_err(int code, char* func)
{
	perror(func);
	exit_program(code);
}
