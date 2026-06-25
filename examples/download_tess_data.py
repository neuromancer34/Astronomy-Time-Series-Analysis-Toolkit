import lightkurve as lk

Data_search = lk.search_lightcurve("RR Lyrae", mission="TESS")
print(Data_search)

if len(Data_search) == 0:
    print("No TESS light curve found for this traget")
else :
    light_curve = Data_search[0].download()
    #print(light_curve)


# Now Coverting this data into a csv file 

df = light_curve.to_pandas()
df = df.reset_index()
df.rename(columns={"index":"time"},inplace=True)
clean_df = df[["time","flux","flux_err"]]
#print(df.index)
#print(clean_df.head())
clean_df.to_csv("lightcurve_dataset.csv",index=False)
