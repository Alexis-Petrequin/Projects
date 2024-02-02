/*
** EPITECH PROJECT, 2021
** init_serv.c
** File description:
** function about initiation
*/

#include "server.h"

static client_t *init_client(void)
{
    client_t *client = malloc(sizeof(client_t) * MAX_CLIENT);

    for (int i = 0; i < MAX_CLIENT; i++) {
        client[i].fd = -1;
        memset(client[i].pseudo, 0, MAX_NAME + 1);
        memset(client[i].id, 0, 37);
        memset(client[i].buffer, 0, (MAX_CMD + 1));
        client[i].logged_in = false;
        memset(client[i].place[0], 0, 37);
        memset(client[i].place[1], 0, 37);
        memset(client[i].place[2], 0, 37);
    }
    return (client);
}

server_t *init_serv(int port)
{
    server_t *server = malloc(sizeof(server_t));

    server->fd_srv = 0;
    server->addr_in.sin_family = AF_INET;
    server->addr_in.sin_addr.s_addr = htonl(INADDR_ANY);
    server->addr_in.sin_port = htons(port);
    server->fd_read = malloc(sizeof(fd_set));
    server->fd_write = malloc(sizeof(fd_set));
    server->max_sd = 0;
    server->client = init_client();
    user_detect();
    return (server);
}

int creat_srv(server_t *server)
{
    int opt = 1;

    if ((server->fd_srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1)
        return (84);
    if (setsockopt(server->fd_srv, SOL_SOCKET, SO_REUSEADDR |
    SO_REUSEPORT, &opt, sizeof(opt)) != 0)
        return (84);
    if (bind(server->fd_srv, (struct sockaddr *) &server->addr_in,
    sizeof server->addr_in) != 0)
        return (84);
    if (listen(server->fd_srv, MAX_CLIENT) != 0)
        return (84);
    return (0);
}