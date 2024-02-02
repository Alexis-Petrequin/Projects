##
## EPITECH PROJECT, 2019
## makefile
## File description:
## makefile
##

SRC	=	create_map.c		\
		check.c				\
		matchstick.c		\
		my_putstr.c			\
		my_number.c			\
		my_putchar.c		\
		errors.c			\
		stickgestion.c		\
		update.c			\
		my_put_nbr.c		\
		iastickgestion.c	\


OBJ	=	$(SRC:.c=.o)

NAME	=	matchstick

all:	$(NAME)

$(NAME):	$(OBJ)
	@gcc -o $(NAME) $(OBJ)

clean:
	@rm -f $(OBJ)

fclean: clean
	@rm -f $(NAME)

re: fclean all

.PHONY: clean fclean
