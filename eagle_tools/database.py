
import eagleSqlTools as sql
import pickle

def query(username,password,query,outfile):

    print('Connecting to EAGLE database...')
    con = sql.connect(username, password=password)
    print('Connected.')

    print('Executing query...')

    data = sql.execute_query(con, query)

    out_data = {}

    for name in data.dtype.names:
        out_data[name] = data[name]

    with open(outfile, 'w') as output:
        pickle.dump(out_data,output)
        output.close()

    print('Dumped output to ',outfile)