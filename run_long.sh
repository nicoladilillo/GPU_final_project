dot_file=(
    "collapse_pyr_dfg__113.txt"
)

area_limit=(
    1000
    500
)

cd result
DFG_folder="../DFGs_new/"

dirlist=($(ls ../source/executable/dfg_v4_1.out))
for f in ${dirlist[@]} 
do 
    for dot in ${dot_file[@]}
    do
        for area in ${area_limit[@]} 
        do
            echo "running $f -- graph $DFG_folder$dot with $area area"
            $f $DFG_folder$dot ../RTL_resoruces.txt $area
        done
    done
done

cd ..
