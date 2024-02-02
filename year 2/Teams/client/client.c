/*
** EPITECH PROJECT, 2021
** client
** File description:
** client function
*/

#include "client.h"

int init_connection(client_t *client, int port, char *ip)
{
    int opt = sizeof(client->addr_in);

    client->fd = 0;
    if ((client->fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1) {
        printf("Init socket fail %s\n", strerror(errno));
        return 84;
    }
    client->addr_in.sin_family = AF_INET;
    client->addr_in.sin_addr.s_addr = inet_addr(ip);
    client->addr_in.sin_port = htons(port);
    if (connect(client->fd, (struct sockaddr *) &client->addr_in,
    sizeof (client->addr_in)) != 0) {
        printf("Connection with the server fail %s\n", strerror(errno));
        return 84;
    }
    return 0;
}

int process(char *buff)
{
    char **args = get_arg(buff);

    if (args == NULL) {
        free(args);
        return (84);
    }
    print_message(args);
    for (int i = 0; args[i]; i++)
        free(args[i]);
    free(args);
    return (0);
}

int parse_recept(char *buff)
{
    int y = 0;
    char buff2[MAX_COM] = { 0 };

    if (buff[0] != '<') {
        printf("%s", buff);
        return (0);
    }
    for (int i = 0; buff[i]; i++) {
        for (y = 0; buff[i] != '\n' && buff[i]; y++, i++)
            buff2[y] = buff[i];
        if (buff2[0] == '<')
            process(buff2);
    }
    return (0);
}

int chat(client_t *client)
{
    char buff[MAX_COM] = { 0 };
    int on = 1;
    int valread = 0;
    fd_set set = { 0 };

    while (on) {
        FD_ZERO(&set);
        FD_SET(STDIN_FILENO, &set);
        FD_SET(client->fd, &set);
        memset(buff, 0, sizeof(buff));
        select(client->fd + 1, &set, NULL, NULL, NULL);
        if (FD_ISSET(STDIN_FILENO, &set)) {
            read(0, buff, MAX_COM);
            dprintf(client->fd, "%s", buff);
        } else if (FD_ISSET(client->fd, &set) &&
        (valread = read(client->fd, buff, sizeof(buff))) > 0) {
            parse_recept(buff);
            memset(buff, 0, MAX_COM);
        }
    }
    return (0);
}

int main(int argc, char **argv)
{
    client_t *client = malloc(sizeof(client_t));

    if (check_errors(argc, argv))
        return end(client, 84);
    if (init_connection(client, atoi(argv[2]), argv[1]))
        return end(client, 84);
    chat(client);
    close(client->fd);
    free(client);
    return end(client, 0);
}