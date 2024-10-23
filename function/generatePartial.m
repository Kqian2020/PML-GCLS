function partial_label = generatePartial(TR_target, average_num)
[num_targte, num_instance] = size(TR_target);
sum_partial = average_num*num_instance;
replace_count = 0;
i = 1;
partial_label = TR_target;
while replace_count < sum_partial
    % first select sum_partial labels
    position_index = randperm(num_targte*num_instance);
    if partial_label(position_index(i)) == -1
        partial_label(position_index(i)) = 1;
        replace_count = replace_count + 1;
    end
    i = i +1;
end
end