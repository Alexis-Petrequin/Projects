/*
** EPITECH PROJECT, 2021
** check_data.c
** File description:
** checking data
*/

#include "server.h"

int check_team_exist(char *uuid)
{
    FILE *fd = fopen(PATH_TM, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char team_id[37] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(uuid, get_data(1, line, team_id)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}

int check_channel_exist(char *uuid)
{
    FILE *fd = fopen(PATH_CH, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char team_id[37] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(uuid, get_data(1, line, team_id)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}

int check_thread_exist(char *uuid)
{
    FILE *fd = fopen(PATH_TH, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char team_id[37] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(uuid, get_data(1, line, team_id)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}

int check_user_exist(char *uuid)
{
    FILE *fd = fopen(PATH_US, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char user_id[37] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(uuid, get_data(0, line, user_id)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}

int check_sub_exist(char *uuid, client_t *client)
{
    char str[50] = { "Database/"};
    FILE *fd = fopen(strcat(str, client->id), "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char user_id[37] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(uuid, get_data(0, line, user_id)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}