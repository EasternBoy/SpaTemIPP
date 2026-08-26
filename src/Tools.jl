function readTime(str::String)

    # starting time: 09-Mar-2022 13:02:32
    secstr = parse(Int64,str[end-1:end]) - 32
    minstr = parse(Int64,str[end-4:end-3]) - 2
    houstr = parse(Int64,str[end-7:end-6]) - 13

    daystr = parse(Int64,str[1:2]) - 9


    return (secstr + minstr*60 + houstr*3600 + daystr*86400)/3600
end