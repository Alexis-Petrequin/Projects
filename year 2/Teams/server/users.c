/*
** EPITECH PROJECT, 2021
** users.c
** File description:
** list users
*/

#include "server.h"

char user_status(server_t *srv, char *uuid)
{
    for (int i = 0; srv->client[i].fd > 0; i++)
        if (strcmp(srv->client[i].id, uuid) == 0)
            return ('1');
    return ('0');
}

int u_users(char **command, client_t *client, server_t *srv)
{
    FILE *fd = fopen(PATH_US, "r");
    char *li = NULL;
    size_t len = 0;
    ssize_t read;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&li, &len, fd)) != -1)
        dprintf(client->fd, "<210><%s><%s><%c>\n",
        get_data(0, li, id), get_data(1, li, name),
        user_status(srv, get_data(0, li, id)));
    if (li)
        free(li);
    fclose(fd);
    return (0);
}

int u_user(char **command, client_t *client, server_t *srv)
{
    FILE *fd = fopen(PATH_US, "r");
    char *li = NULL;
    size_t len = 0;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };

    if (fd == NULL)
        return (84);
    if (check_user_exist(command[1]))
        return (dprintf(client->fd, "<513><%s>\n", command[1]));
    while (getline(&li, &len, fd) != -1)
        if (strcmp(get_data(0, li, id), command[1]) == 0)
            dprintf(client->fd, "<310><%s><%s><%c>\n", id,
            get_data(1, li, name), user_status(srv, id));
    fclose(fd);
    return (0);
}