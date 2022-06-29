dot_file=(
    "arf.txt"
    "fir.txt"
    "ewf.txt"
    "h2v2_smooth_downsample_dfg__6.txt"
)

area_limit=(
    1000
    500
)

cd source
make all
cd ../result
DFG_folder="../DFGs_new/"

dirlist=($(ls ../source/executable/*.out))
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

mv *.log log_file
cd ..
