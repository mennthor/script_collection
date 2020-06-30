import datetime


TEMPLATE = """### Week KW{0:}

#### {1:}-{2:02d}-{7:02d} (Monday)
_NOENTRY_

#### {1:}-{3:02d}-{8:02d} (Tuesday)
_NOENTRY_

#### {1:}-{4:02d}-{9:02d} (Wednesday)
_NOENTRY_

#### {1:}-{5:02d}-{10:02d} (Thursday)
_NOENTRY_

#### {1:}-{6:02d}-{11:02d} (Friday)
_NOENTRY_"""


def main():
    today = datetime.date.today()
    year, week, _ = today.isocalendar()
    # From: https://stackoverflow.com/questions/17277002
    # Monday to Friday
    weekdates = [today + datetime.timedelta(days=i)
                 for i in range(0 - today.weekday(), 5 - today.weekday())]
    months = [wd.month for wd in weekdates]
    days = [wd.day for wd in weekdates]
    insert = months + days

    print(TEMPLATE.format(week, year, *insert))


if __name__ == "__main__":
    main()
