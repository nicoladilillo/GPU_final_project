dot_file=(
    "arf.txt"
    "fir.txt"
    "ewf.txt"
)

area_limit=(
    500
)

cd source
make all
cd ../result
DFG_folder="../DFGs_new/"

dirlist=($(ls ../source/executable/*CPU.out))
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
