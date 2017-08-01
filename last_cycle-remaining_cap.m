for i = 1:numel(batch2)
    batch2(i).last_cycle = numel(batch2(i).cycles);
end

for i = 1:numel(batch2)
    last_cycle = batch2(i).last_cycle;
    batch2(i).rem_cap = batch2(i).cycles(last_cycle).Qd(end);
end


for i = 1:numel(batch1)
    batch1(i).last_cycle = numel(batch1(i).cycles);
end

for i = 1:numel(batch1)
    last_cycle = batch1(i).last_cycle;
    batch1(i).rem_cap = batch1(i).cycles(last_cycle).Qd(end);
end

for i = 1:numel(batch2_heldout)
    batch2_heldout(i).last_cycle = numel(batch1(i).cycles) + numel(batch2_heldout(i).cycles);
end