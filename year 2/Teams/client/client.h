/*
** EPITECH PROJECT, 2021
** client.h
** File description:
** header
*/

#ifndef CLIENT_H_
#define CLIENT_H_

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "../libs/myteams/logging_client.h"
#include <math.h>
#define MAX_COM 1200

typedef struct client_s {
    int fd;
    struct sockaddr_in addr_in;
} client_t;

int print_message4(char **args);
int print_message3(char **args);
int print_message2(char **args);
int print_message(char **args);
int check_errors(int argc, char **argv);
int end(client_t *client, int ret);
int get_nbarg(char *buff);
char **get_arg(char *buff);

#endif
