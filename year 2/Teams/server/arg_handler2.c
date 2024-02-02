/*
** EPITECH PROJECT, 2021
** arg_handler2.c
** File description:
** abouter handling args
*/

#include "server.h"

int handle_buffer(char *buff, client_t *client)
{
    if (strlen(buff) > (MAX_CMD)) {
        dprintf(client->fd, "<520>\n");
        return (84);
    }
    if (strchr(buff, '\n') == NULL) {
        dprintf(client->fd, "<520>\n");
        return (84);
    }
    strcat(client->buffer, buff);
    return (0);
}

int create_thread2(char *name, client_t cl, char *body, FILE *fd)
{
    if (!check_thread_name(name)) {
        fclose(fd);
        dprintf(cl.fd, "<530>\n");
        return (84);
    }
    if ((strlen(name) > MAX_NAME) || strlen(body) > MAX_BODY) {
        fclose(fd);
        return (84);
    }
    return (0);
}

void help(void)
{
    printf("USAGE: ./myteams_server port\n\n\tport is the port"
    " number on which the server socket listens.\n");
}

int check_errs(int argc, char** argv)
{
    if (argc != 2)
        return (84);
    if (!strcmp("-help", argv[1])) {
        help();
        return (84);
    }
    for (int i = 0; argv[1][i]; ++i)
        if (argv[1][i] < '0' || argv[1][i] > '9')
            return (84);
    return (0);
}