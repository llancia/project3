import recsystem as rc
data = rc.load_data("ratings.dat", "::")
train, test = rc.split_test_train(data, 0.8)
M = rc.create_matrix(train)
print "loaded"
useravg, itemavg = rc.find_user_and_item_avg(M)
new_U, new_V = rc.my_train_SVD(M)
print rc.funk_validate_with_testdata(new_U,new_V,test, useravg)
