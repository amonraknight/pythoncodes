import pandas as pd
from datetime import datetime, date

#requires matplotlib

df = pd.DataFrame({'Date and time': [datetime(2015, 1, 1, 11, 30, 55),
                                     datetime(2015, 1, 2, 1, 20, 33),
                                     datetime(2015, 1, 3, 11, 10),
                                     datetime(2015, 1, 4, 16, 45, 35),
                                     datetime(2015, 1, 5, 12, 10, 15)],
                   'Dates only': [date(2015, 2, 1),
                                  date(2015, 2, 2),
                                  date(2015, 2, 3),
                                  date(2015, 2, 4),
                                  date(2015, 2, 5)],
                   'Numbers': [1010, 2020, 3030, 2020, 1515],
                   'Percentage': [.1, .2, .33, .25, .5],
                   })
df['final'] = [f"=C{i}*D{i}" for i in range(2, df.shape[0]+2)]

df_style = df.style.applymap(lambda x: 'color:red', subset=["Date and time"]) \
    .applymap(lambda x: 'color:green', subset=["Dates only"]) \
    .applymap(lambda x: 'background-color:#ADD8E6', subset=["Numbers"]) \
    .background_gradient(cmap="PuBu", low=0, high=0.5, subset=["Percentage"])

writer = pd.ExcelWriter("demo_style.xlsx",
                        datetime_format='mmm d yyyy hh:mm:ss',
                        date_format='mmmm dd yyyy')
df_style.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()