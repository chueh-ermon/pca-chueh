for i = 1:numel(batch)
    batch(i).last_cycle = numel(batch(i).cycles);
end

for i = 1:numel(batch)
    last_cycle = batch(i).last_cycle;
    batch(i).rem_cap = batch(i).cycles(last_cycle).Qd(end);
end