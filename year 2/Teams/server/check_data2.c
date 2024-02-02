/*
** EPITECH PROJECT, 2021
** check_data2.c
** File description:
** checking data
*/

#include "server.h"

int check_team_name(char *name)
{
    FILE *fd = fopen(PATH_TM, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char name2[MAX_NAME + 1] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(name, get_data(0, line, name2)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}

int check_channel_name(char *name)
{
    FILE *fd = fopen(PATH_CH, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char name2[MAX_NAME + 1] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(name, get_data(0, line, name2)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}

int check_thread_name(char *name)
{
    FILE *fd = fopen(PATH_TH, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char name2[MAX_NAME + 1] = { 0 };

    if (fd == NULL)
        return (84);
    while ((read = getline(&line, &len, fd)) != -1)
        if (strcmp(name, get_data(0, line, name2)) == 0) {
            fclose(fd);
            return (0);
        }
    fclose(fd);
    return (84);
}