import sys 
import logging


def error(msg,msg_detail:sys):
    _,_,exc_tb = msg_detail.exc_info()
    Error_Message = f"The Error is : {str(msg)} \nLine_No: {exc_tb.tb_lineno}"

    return Error_Message


class CustomException(Exception):
    def __init__(self, Error_Message, msg_detail:sys):
        super().__init__(Error_Message) # this will print the error message
        self.Error_Message = error(Error_Message,msg_detail) # this will print the error message with line number

    def __str__(self):
        return self.Error_Message
    

'''
just to check working of the exception 
if __name__ == "__main__":
    try:
        X = 1/0
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)
'''
