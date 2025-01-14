import logging


def log(msg, level="error"):
    """Logs a message with the specified log level.

    Args:
        msg (str): The message to log.
        level (str, optional): The log level. Can be one of `debug`, `info`, `warning`, or `error`. Defaults to `error`.
    """
    match level:
        case "debug":
            level = logging.DEBUG
        case "info":
            level = logging.INFO
        case "warning":
            level = logging.WARNING
        case "error":
            level = logging.ERROR
        case _:
            level = logging.ERROR

    logging.basicConfig(
        filename="conversational-highlights.log",
        level=level,
        format="%(asctime)s : %(levelname)s : %(message)s",
    )

    logging.exception(msg)


def get_default_text():
    """Returns the default text to be used in the summarization process.

    Returns:
        str: The default text.
    """
    return """
Customer: Hi there I have received a message saying that a representative will be sent to my home regarding bills could I please talk to someone that can help. Thanks.. 
Agent: Thank you for choosing to chat with us on WhatsApp! To continue this conversation, please can you confirm your full name, address including your postcode and your date of birth?. 
Customer: Hi yes . 
Agent: Thank you, a member of the team will reply to you shortly.. 
Agent: Hi there , so we just need to set an arrangement on you account today to prevent any further action. 
Customer: Ok what does that mean?. 
Agent: You would need to cover your usage as a minimum at £32 monthly. 
Agent: We need a payment plan in place. 
Agent: So this won't go down the representative route. 
Customer: Ok I just want to apologise and let you know that I have been going through very hard times. My partner got cancer so I was taking care of him then he unfortunately died and I have been on my own since and not been coping too well. How do I go about sorting this please.. 
Agent: I am very sorry to hear this, i can imagine this is a very difficult time for you. 
Agent: We can get something set up now for you, we can do a monthly payment plan?. 
Customer: Ok thank you so how much would I be paying in total please. 
Agent: In total as a minimum you would need to cover your usage at £33 monthly, anything on top of this will go towards the arrears. 
Customer: So will I be charged more than £33 monthly then?. 
Agent: Yes, that is your usage. 
Customer: So how much will I be charged then?. 
Agent: Your usage at £33 monthly. 
Agent: You can pay more to cover some of the arrears but we can't accept anything less than £33 monthly. 
Customer: So how do I go about paying the arrears then? How much arrears do I need to pay.. 
Customer: I mean what is the total arrears please.. 
Agent: The total is £3702.74. 
Customer: So could we do £35 monthly for now as I am also permanently signed off from work now also due to ill health.. 
Agent: We can do this, have you ever considered having a meter and spoken about this to Essex  and Suffolk who would provide this for you?. 
Customer: No but I would consider it and would be grateful to speak with someone who can help me.. 
Agent: So you would need to call Essex and Suffolk water on 03457 820 111 and they can help with this, in the meantime, i am happy to set up £35 monthly for your sewerage?. 
Customer: Yes please and thank you. 
Agent: Perfect, how would you like to pay? and what dates, the 1st, 8th, 15th, 15th. 
Customer: By card please can we do 1st of the month please. 
Agent: Is that payment card or direct debit? and we do offer the 1st. 
Customer: Direct debit please. 
Agent: Can you send in two separate messages your account number and sort code please?. 
Customer: Account number is. 
Customer: Sort code is . 
Agent: Perfect, this is set for you, first payment is due on the 1st Aug, have sent a confirmation text. 
Agent: I must make you aware, due to the balance, it will have a negative impact on your credit rating, however, it will show you have an arrangement. 
Customer: Ok thank you ever so much for all the help. 
Agent: You're more than welcome, can I help with anything else at all? 
Customer: No that's it for now ta very much indeed.
    """
