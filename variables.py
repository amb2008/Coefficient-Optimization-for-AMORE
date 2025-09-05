# mech file is the name of the reduced mechanism that you would like to optimize
# full eqn is the path to the file that contains the full mechanism equation
# full spc is the path to the file that contains all the species
# input conditions are what conditions the mechanism will be simulated at. Set it to 1 if you want default conditions, or otherwise set it as a custom list.
# individual params I may never use
# ignore coeffs are the coefficients you don't want to optimize. To ignore all coeffs that are 1.0, set ignore as "ones." To only ignore all coeffs that are isop, set ignore as "isop." To ignore both, set ignore as "isop_ones." To ignore coefficients on a customized case to case basis, make a list of the same structure as prod_coeffs_list_r, and at the index of coefficients you do want to optimize put a 1 in ignore, and for ones you don't want to optimize put a 0.

#config = {'mech_file': './reduced_mech.txt', 'full_eqn': './caltech_amore_isoprene_full_eqn.txt', 'full_spc': './caltech_amore_isoprene_full_spc.txt', 'input_conditions': 1, 'individual_params': 2, 'ignore': "ones"}

config = {'mech_file': './reduced_mech.txt', 'full_eqn': './caltech_amore_isoprene_full_eqn.txt', 'full_spc': './caltech_amore_isoprene_full_spc.txt', 'input_conditions': 1, 'individual_params': 2, 'ignore': "nothing"}
