/*
** EPITECH PROJECT, 2021
** event.c
** File description:
** about connection event
*/

#include "server.h"

static int available(client_t *client)
{
    int i = 0;

    for (; i < MAX_CLIENT; ++i)
        if (client[i].fd == -1)
            return (i);
    return (MAX_CLIENT);
}

int new_connection(server_t *srv)
{
    struct sockaddr_in addr = { 0 };
    int len = sizeof(addr);
    int i = available(srv->client);
    uuid_t id;
    char t_uuid[37] = { 0 };

    if (i == MAX_CLIENT)
        return (0);
    if ((srv->client[i].fd = accept(srv->fd_srv, (struct sockaddr *) &addr,
    (socklen_t*) &len)) != -1) {
        uuid_generate_random(id);
        uuid_unparse_upper(id, srv->client[i].id);
    } else
        return (84);
    return (0);
}

void connection_lost(server_t *srv, int i)
{
    server_event_user_logged_out(srv->client[i].id);
    close(srv->client[i].fd);
    srv->client[i].fd = -1;
    srv->client[i].logged_in = false;
    memset(srv->client[i].id, 0, 37);
    memset(srv->client[i].buffer, 0, (MAX_CMD * 2 + 1));
    memset(srv->client[i].pseudo, 0, MAX_NAME);
}

int u_logout(char **command, client_t *client, server_t *srv)
{
    server_event_user_logged_out(client->id);
    for (int i = 0; srv->client[i].fd != -1; i++)
        dprintf(srv->client[i].fd, "<120><%s><%s>\n",
        client->id, client->pseudo);
    close(client->fd);
    client->fd = -1;
    client->logged_in = false;
    memset(client->id, 0, 37);
    memset(client->buffer, 0, (MAX_CMD * 2 + 1));
    memset(client->pseudo, 0, MAX_NAME);
    return (0);
}