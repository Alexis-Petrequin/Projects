/*
** EPITECH PROJECT, 2021
** list.c
** File description:
** list data
*/

#include "server.h"

int list_team(client_t *client)
{
    FILE *fd = fopen(PATH_TM, "r");
    char *li = NULL;
    size_t len = 0;
    ssize_t read;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };
    char desc[MAX_DESCRIPTION + 1] = { 0 };

    while ((read = getline(&li, &len, fd)) != -1)
        dprintf(client->fd, "<211><%s><%s><%s>\n",get_data(1, li, id),
            get_data(0, li, name), get_data(3, li, desc));
    fclose(fd);
    return (0);
}

int list_channel(client_t *client)
{
    FILE *fd = fopen(PATH_CH, "r");
    char *li = NULL;
    size_t len = 0;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };
    char desc[MAX_DESCRIPTION + 1] = { 0 };

    while (getline(&li, &len, fd) != -1)
        if (strcmp(client->place[0], get_data(2, li, id)) == 0)
            dprintf(client->fd, "<212><%s><%s><%s>\n",get_data(1, li, id),
            get_data(0, li, name), get_data(3, li, desc));
    fclose(fd);
    return (0);
}

int list_thread(client_t *client)
{
    FILE *fd = fopen(PATH_TH, "r");
    char *li = NULL;
    size_t len = 0;
    ssize_t read;
    char id[37] = { 0 };
    char cid[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };
    char time[15] = { 0 };
    char body[MAX_BODY + 1] = { 0 };

    while ((read = getline(&li, &len, fd)) != -1)
        if (strcmp(client->place[1], get_data(3, li, id)) == 0)
            dprintf(client->fd, "<213><%s><%s><%s><%s><%s>\n",
            get_data(1, li, id), get_data(4, li, cid), get_data(6, li, time),
            get_data(0, li, name), get_data(5, li, body));
    fclose(fd);
    return (0);
}

int list_reply(client_t *client)
{
    FILE *fd = fopen(PATH_RE, "r");
    char *li = NULL;
    size_t len = 0;
    ssize_t read;
    char id[37] = { 0 };
    char cid[37] = { 0 };
    char tim[15] = { 0 };
    char body[MAX_BODY + 1] = { 0 };

    while ((read = getline(&li, &len, fd)) != -1)
        if (strcmp(client->place[2], get_data(2, li, id)) == 0)
            dprintf(client->fd, "<213><%s><%s><%s><%s>\n",
            get_data(2, li, id), get_data(3, li, cid), get_data(5, li, tim),
            get_data(4, li, body));
    fclose(fd);
    return (0);
}

int u_list(char **command, client_t *client, server_t *srv)
{
    if (strcmp(client->place[0], "") == 0)
        return (list_team(client));
    if (check_team_exist(client->place[0]))
        return (dprintf(client->fd, "<510><%s>\n", client->place[0]));
    if (strcmp(client->place[1], "") == 0)
        return (list_channel(client));
    if (check_channel_exist(client->place[1]))
        return (dprintf(client->fd, "<511><%s>\n", client->place[1]));
    if (strcmp(client->place[2], "") == 0)
        return (list_thread(client));
    if (check_thread_exist(client->place[2]))
        return (dprintf(client->fd, "<512><%s>\n", client->place[2]));
    if (strcmp(client->place[2], "") != 0)
        return (list_reply(client));
    return (0);
}