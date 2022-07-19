import re

def check(t, sub='Conv'):
    flag = False
    try:
        s = re.findall("['](.*)[']", str(t))[0]
        s = s.split('.')
        s = set(s)
    except Exception as e:
        print(f'Something wrong with {e}')
    else:
        if isinstance(sub,str):
            if sub in s:
                flag = True
        elif isinstance(sub,list):
           if set(sub)  & s:
               flag = True
    return flag

