def hypenAdd(s):
    # s = '56JTT5'
    # print(type(strs[0]))
    if(len(s) !=6):
        return s
    #zd-pb-67
    if(s[0].isalpha() and s[1].isalpha() and s[2].isalpha() and s[3].isalpha() and s[4].isdigit() and s[5].isdigit()):
        rs = s[0:2] + '-' + s[2:4]+ '-'+s[4:]
        # print(rs)
        return rs
    #67-fp-sj
    elif (s[0].isdigit() and s[1].isdigit() and s[2].isalpha() and s[3].isalpha() and s[4].isalpha() and s[5].isalpha()):
        rs = s[0:2] + '-' + s[2:4] + '-' + s[4:]
        # print(rs)
        return rs
    #vd-020-p
    elif (s[0].isalpha() and s[1].isalpha() and s[2].isdigit() and s[3].isdigit() and s[4].isdigit() and s[5].isalpha()):
        rs = s[0:2] + '-' + s[2:5] + '-' + s[5:]
        return rs
    #v-599-rl
    elif (s[0].isalpha() and s[1].isdigit() and s[2].isdigit() and s[3].isdigit() and s[4].isalpha() and s[4].isalpha()):
        rs = s[0] + '-' + s[1:4] + '-' + s[4:]
        return rs
    #3-vfx-39
    elif (s[0].isdigit() and s[4].isdigit() and s[5].isdigit() and s[1].isalpha() and s[2].isalpha() and s[3].isalpha()):
        rs = s[0] + '-' + s[1:4] + '-' + s[4:]
        # print(rs)
        return rs
    #56-jtt-5
    elif (s[0].isdigit() and s[1].isdigit() and s[5].isdigit() and s[2].isalpha() and s[3].isalpha() and s[4].isalpha()):
        rs = s[0:2] + '-' + s[2:5] + '-' + s[5:]
        # print(rs)
        return rs
    else:
        # print(s)
        return s

if __name__ == "__main__":
    s = 'V599RL'
    rs = hypenAdd(s)
    print(rs)