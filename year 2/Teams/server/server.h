/*
** EPITECH PROJECT, 2021
** server.h
** File description:
** server head
*/

#ifndef SERVER_H_
#define SERVER_H_

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <uuid/uuid.h>
#include <fcntl.h>
#include <time.h>
#include "../libs/myteams/logging_server.h"

#define MAX_CLIENT 20
#define MAX_CMD 1024
#define MAX_NAME 32
#define MAX_DESCRIPTION 255
#define MAX_BODY 512
#define PATH_TM "Database/Teams.txt"
#define PATH_CH "Database/Channels.txt"
#define PATH_TH "Database/Threads.txt"
#define PATH_RE "Database/Replies.txt"
#define PATH_US "Database/Users.txt"


typedef struct client_s {
    int fd;
    char id[37];
    char pseudo[MAX_NAME + 1];
    char buffer[MAX_CMD + 1];
    bool logged_in;
    char place[3][37];
} client_t;

typedef struct server_s {
    int fd_srv;
    struct sockaddr_in addr_in;
    fd_set *fd_read;
    fd_set *fd_write;
    int max_sd;
    client_t *client;
} server_t;

int new_connection(server_t *srv);
void connection_lost(server_t *srv, int i);
server_t *init_serv(int port);
int creat_srv(server_t *server);
char *get_data(int i, char *line, char *data);
int u_login(char **command, client_t *client, server_t *srv);
int user_loging(char *name, client_t *client);
void user_detect(void);
char *get_data(int i, char *line, char *data);
int check_team_exist(char *uuid);
int check_thread_exist(char *uuid);
int check_channel_exist(char *uuid);
int check_user_exist(char *uuid);
int check_sub_exist(char *uuid, client_t *client);
int u_unsub(char **command, client_t *client, server_t *srv);
int u_sub(char **command, client_t *client, server_t *srv);
int handle_buffer(char *buff, client_t *client);
int check_args_function(char *buffer, int nb);
int parse_buffer(char *str, char ***map);
int u_logout(char **command, client_t *client, server_t *srv);
int u_create(char **command, client_t *client, server_t *srv);
int u_users(char **command, client_t *client, server_t *srv);
int u_user(char **command, client_t *client, server_t *srv);
int u_list(char **command, client_t *client, server_t *srv);
int u_help(char **command, client_t *client, server_t *srv);
int u_info(char **command, client_t *client, server_t *srv);
int u_subscribed(char **command, client_t *client, server_t *srv);
int u_use(char **command, client_t *client, server_t *srv);
int u_send(char **command, client_t *client, server_t *srv);
int u_messages(char **command, client_t *client, server_t *srv);
char user_status(server_t *srv, char *uuid);
int check_thread_name(char *name);
int check_channel_name(char *name);
int check_team_name(char *name);
int create_thread2(char *name, client_t cl, char *body, FILE *fd);
int check_errs(int argc, char** argv);

#endif
