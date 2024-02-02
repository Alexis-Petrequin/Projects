##
## EPITECH PROJECT, 2021
## makefile
## File description:
## makefile
##

SRC	=	client.c 		\
		print_msg.c 	\
		utils.c 		\

OBJ	=	$(SRC:.c=.o)

NAME	=	myteams_cli

all:	$(NAME)

$(NAME):	$(OBJ)
	@gcc -o $(NAME) $(OBJ) -Wall -Wextra -I../libs/myteams -L ../libs/myteams -lmyteams -luuid
	cp myteams_cli ../

clean:
	@rm -f $(OBJ)

fclean: clean
	@rm -f $(NAME)
	@rm -f ../$(NAME)

re: fclean all

.PHONY: clean fclean
