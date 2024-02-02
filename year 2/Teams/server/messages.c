/*
** EPITECH PROJECT, 2021
** messages.c
** File description:
** about function msg
*/

#include "server.h"

int send_msg(char **command, client_t *client, server_t *srv)
{
    char str[50] = { "Database/"};
    char str2[50] = { "Database/"};
    FILE *fd = fopen(strcat(str2, command[1]), "a+");
    FILE *fd2 = fopen(strcat(str, client->id), "a+");
    time_t tim;

    server_event_private_message_sended(client->id, command[1], command[2]);
    fprintf(fd, "%s;%s;%s;%ld;\n", client->id, command[1],
    command[2], time(&tim));
    fprintf(fd2, "%s;%s;%s;%ld;\n", client->id, command[1],
    command[2], time(&tim));
    fclose(fd);
    fclose(fd2);
    return (0);
}

int u_send(char **command, client_t *client, server_t *srv)
{
    if (check_user_exist(command[1]))
        return (dprintf(client->fd, "<513><%s>\n", command[1]));
    if (strlen(command[2]) > MAX_BODY) {
        dprintf(client->fd, "<520>\n");
        return (84);
    }
    send_msg(command, client, srv);
    for (int i = 0; srv->client[i].fd > 0; i++)
        if (strcmp(srv->client[i].id, command[1]) == 0)
            dprintf(srv->client[i].fd, "<130><%s><%s>\n",
            client->id, command[2]);
    return (0);
}

int list_messages2(char **command, client_t *client, char *li)
{
    char id[37] = { 0 };
    char id2[37] = { 0 };
    char tim[15] = { 0 };
    char body[MAX_BODY + 1] = { 0 };

    if ((strcmp(command[1], get_data(0, li, id)) == 0 ||
        strcmp(client->id, get_data(0, li, id)) == 0) &&
        (strcmp(client->id, get_data(1, li, id2)) == 0 ||
        strcmp(command[1], get_data(1, li, id2)) == 0))
        return (dprintf(client->fd, "<215><%s><%s><%s>\n",
        get_data(0, li, id), get_data(3, li, tim), get_data(2, li, body)));
    return (0);
}

int list_messages(char **command, client_t *client)
{
    char str[50] = { "Database/"};
    FILE *fd = fopen(strcat(str, client->id), "r");
    char *li = NULL;
    size_t len = 0;

    if (fd == NULL)
        return (84);
    while (getline(&li, &len, fd) != -1)
        list_messages2(command, client, li);
    fclose(fd);
    return (0);
}

int u_messages(char **command, client_t *client, server_t *srv)
{
    if (check_user_exist(command[1]))
        return (dprintf(client->fd, "<513><%s>\n", command[1]));
    return (list_messages(command, client));
}