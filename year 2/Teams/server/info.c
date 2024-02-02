/*
** EPITECH PROJECT, 2021
** info.c
** File description:
** about info cmd
*/

#include "server.h"

int info_user(client_t *client)
{
    char temp = '0';

    if (client->logged_in == true)
        temp = '1';
    dprintf(client->fd, "<310><%s><%s><%c>\n",
    client->id, client->pseudo, temp);
    return (0);
}

int info_team(client_t *client)
{
    FILE *fd = fopen(PATH_TM, "r");
    char *li = NULL;
    size_t len = 0;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };
    char descr[MAX_DESCRIPTION + 1] = { 0 };

    while (getline(&li, &len, fd) != -1)
        if (strcmp(client->place[0], get_data(1, li, id)) == 0) {
            dprintf(client->fd, "<311><%s><%s><%s>\n", get_data(1, li, id),
            get_data(0, li, name), get_data(3, li, descr));
            break;
        }
    fclose(fd);
    return (0);
}

int info_channel(client_t *client)
{
    FILE *fd = fopen(PATH_CH, "r");
    char *li = NULL;
    size_t len = 0;
    char id[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };
    char descr[MAX_DESCRIPTION + 1] = { 0 };

    while (getline(&li, &len, fd) != -1)
        if (strcmp(client->place[1], get_data(1, li, id)) == 0) {
            dprintf(client->fd, "<312><%s><%s><%s>\n", get_data(1, li, id),
            get_data(0, li, name), get_data(3, li, descr));
            break;
        }
    fclose(fd);
    return (0);
}

int info_thread(client_t *client)
{
    FILE *fd = fopen(PATH_TH, "r");
    char *li = NULL;
    size_t len = 0;
    char id[37] = { 0 };
    char id2[37] = { 0 };
    char name[MAX_NAME + 1] = { 0 };
    char descr[MAX_BODY + 1] = { 0 };
    char tim[15] = { 0 };

    while (getline(&li, &len, fd) != -1)
        if (strcmp(client->place[2], get_data(1, li, id)) == 0) {
            dprintf(client->fd, "<313><%s><%s><%s><%s><%s>\n",
            get_data(1, li, id), get_data(4, li, id2), get_data(6, li, tim),
            get_data(0, li, name), get_data(5, li, descr));
            break;
        }
    fclose(fd);
    return (0);
}

int u_info(char **command, client_t *client, server_t *srv)
{
    if (strcmp(client->place[0], "") == 0)
        return (info_user(client));
    if (check_team_exist(client->place[0]))
        return (dprintf(client->fd, "<510><%s>\n", client->place[0]));
    if (strcmp(client->place[1], "") == 0)
        return (info_team(client));
    if (check_channel_exist(client->place[1]))
        return (dprintf(client->fd, "<511><%s>\n", client->place[1]));
    if (strcmp(client->place[2], "") == 0)
        return (info_channel(client));
    if (check_thread_exist(client->place[2]))
        return (dprintf(client->fd, "<512><%s>\n", client->place[2]));
    if (strcmp(client->place[2], "") != 0)
        return (info_thread(client));
    return (0);
}