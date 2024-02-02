/*
** EPITECH PROJECT, 2021
** createcom.c
** File description:
** function about create
*/

#include "server.h"

int create_team(char *name, char *descr, client_t client, server_t *srv)
{
    FILE *fd = fopen(PATH_TM, "a+");
    uuid_t id;
    char t_uuid[37] = { 0 };

    if (!check_team_name(name))
        return (dprintf(client.fd, "<530>\n"));
    if ((strlen(name) > MAX_NAME) || strlen(descr) > MAX_DESCRIPTION) {
        fclose(fd);
        return (84);
    }
    uuid_generate_random(id);
    uuid_unparse_upper(id, t_uuid);
    dprintf(client.fd, "<410><%s><%s><%s>\n", t_uuid, name, descr);
    fprintf(fd, "%s;%s;%s;%s;\n", name, t_uuid, client.id, descr);
    server_event_team_created(t_uuid, name, client.id);
    for (int i = 0; srv->client[i].fd > 0; i++)
        dprintf(srv->client[i].fd, "<140><%s><%s><%s>\n",
        t_uuid, name, descr);
    fclose(fd);
    return (0);
}

int create_channel(char *name, char *des, client_t client, server_t *srv)
{
    FILE *fd = fopen(PATH_CH, "a+");
    uuid_t id;
    char uid[37] = { 0 };

    if (!check_channel_name(name))
        return (dprintf(client.fd, "<530>\n"));
    if ((strlen(name) > MAX_NAME) || strlen(des) > MAX_DESCRIPTION) {
        fclose(fd);
        return (84);
    }
    uuid_generate_random(id);
    uuid_unparse_upper(id, uid);
    dprintf(client.fd, "<411><%s><%s><%s>\n", uid, name, des);
    fprintf(fd, "%s;%s;%s;%s;\n", name, uid, client.place[0], des);
    server_event_channel_created(client.place[0], uid, name);
    for (int i = 0; srv->client[i].fd > 0; i++)
        if (srv->client[i].logged_in == true &&
        check_sub_exist(client.place[0], &srv->client[i]) == 0)
            dprintf(srv->client[i].fd, "<141><%s><%s><%s>\n", uid, name, des);
    fclose(fd);
    return (0);
}

int create_thread(char *name, char *body, client_t cl, server_t *srv)
{
    FILE *fd = fopen(PATH_TH, "a+");
    uuid_t id;
    char ud[37] = { 0 };
    time_t tim;
    time_t tp = time(&tim);

    create_thread2(name, cl, body, fd);
    uuid_generate_random(id);
    uuid_unparse_upper(id, ud);
    dprintf(cl.fd, "<412><%s><%s><%ld><%s><%s>\n", ud, cl.id, tp, name, body);
    fprintf(fd, "%s;%s;%s;%s;%s;%s;%ld;\n",
    name, ud, cl.place[0], cl.place[1], cl.id, body, tp);
    server_event_thread_created(cl.place[1], ud, cl.id, name, body);
    for (int i = 0; srv->client[i].fd > 0; i++)
        if (srv->client[i].logged_in == true &&
        check_sub_exist(cl.place[0], &srv->client[i]) == 0)
            dprintf(srv->client[i].fd, "<142><%s><%s><%ld><%s><%s>\n",
            ud, cl.id, tp, name, body);
    fclose(fd);
    return (0);
}

int create_reply(char *body, client_t client, server_t *srv)
{
    FILE *fd = fopen(PATH_RE, "a+");
    time_t tim;
    time_t temp = time(&tim);

    if (strlen(body) > MAX_BODY) {
        fclose(fd);
        return (84);
    }
    dprintf(client.fd, "<413><%s><%s><%ld><%s>\n",
    client.place[2], client.id, temp, body);
    fprintf(fd, "%s;%s;%s;%s;%s;%ld;\n",
    client.place[0], client.place[1], client.place[2], client.id, body, temp);
    server_event_reply_created(client.place[2], client.id, body);
    for (int i = 0; srv->client[i].fd > 0; i++)
        if (srv->client[i].logged_in == true &&
        check_sub_exist(client.place[0], &srv->client[i]) == 0)
            dprintf(srv->client[i].fd, "<143><%s><%s><%s><%s>\n",
            client.place[0], client.place[2], client.id, body);
    fclose(fd);
    return (0);
}

int u_create(char **command, client_t *client, server_t *srv)
{
    if (command[2] != NULL) {
        if (strcmp(client->place[0], "") == 0)
            return (create_team(command[1], command[2], *client, srv));
        if (check_team_exist(client->place[0]))
            return (dprintf(client->fd, "<510><%s>\n", client->place[0]));
        if (strcmp(client->place[1], "") == 0)
            return (create_channel(command[1], command[2], *client, srv));
        if (check_channel_exist(client->place[1]))
            return (dprintf(client->fd, "<511><%s>\n", client->place[1]));
        if (strcmp(client->place[2], "") == 0)
            return (create_thread(command[1], command[2], *client, srv));
    }
    if (check_team_exist(client->place[0]))
        return (dprintf(client->fd, "<510><%s>\n", client->place[0]));
    if (check_channel_exist(client->place[1]))
        return (dprintf(client->fd, "<511><%s>\n", client->place[1]));
    if (check_thread_exist(client->place[2]))
        return (dprintf(client->fd, "<512><%s>\n", client->place[2]));
    if (strcmp(client->place[2], "") != 0 && command[2] == NULL)
            return (create_reply(command[1], *client, srv));
    return (dprintf(client->fd, "<520>\n"));
}