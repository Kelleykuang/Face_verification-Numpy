threh = (-20:0.01:20)/100;
score = zeros(size(threh));
for i = 1:max(size(threh))
    tp = 0;
    fp = 0;
    tn = 0;
    fn = 0;
    for j = 1:size(match)
        if(match(j)>=threh(i))
            tp = tp + 1;
        else
            fn = fn + 1;
        end
    end
    for j = 1:size(mis)
        if(mis(j)>=threh(i))
            fp = fp + 1;
        else
            tn = tn + 1;
        end
    end
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f1 = 2/(1/precision+1/recall);
    score(i) = f1;
end
max(score)
plot(score);