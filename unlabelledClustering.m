% Ensure that data for F32 is loaded
data = load('feat.mat');
F32 = data.f32;

% Pick 10 random points as initial prototypes
prototypeInd = randsample(160, 10);
prototypes = zeros(2,10);
for i = 1:10
    prototypes(:,i) = F32(1:2,prototypeInd(i));
end

% Calculate new prototypes via one iteration of K-Means algorithms
newPrototype = kMeans(F32, prototypes);

% While the old prototype does not equal the new one, continue to iterate
% through K-Means algorithm
iteration = 1;
while ~isequal(prototypes, newPrototype)
    oldPrototype = newPrototype;
    newPrototype = kMeans(F32, oldPrototype);
    prototypes = oldPrototype;
    iteration = iteration + 1;
    iteration
end

% Plot original data and final, converged prototypes
figure
scatter(F32(1,:), F32(2,:), 30);
hold on
scatter(newPrototype(1,:), newPrototype(2,:), 'rx', 'linewidth', 3);
legend('Unlabelled Data Points', 'Converged Prototype Means', 'Location', 'northwest')
xlabel('xij(1)')
ylabel('xij(2)')
title(sprintf('Converged Protoypes via K-means Algorithm (Iterations: %d)', iteration))

% Function that represents an iteration of the K-Means algorithm
% Inputs: 
%         F32: original dataset
%         prototypes: old means of prototypes
% Outputs: 
%         newMeans: new means of prototypes
function newMeans = kMeans(F32, prototypes)
    
    % Calculate the Euclidean distance between for each data point (j) and
    % each prototype mean (i)
    distanceArray = zeros(10,160);
    for i = 1:10
        for j = 1:160
            distanceArray(i,j) = (prototypes(1,i)-F32(1,j))^2 + (prototypes(2,i)-F32(2,j))^2;
        end
    end
    
    % Determine the index of the closest class by taking minimum Euclidean
    % distance, and classify points
    closestClass = zeros(3,160);
    for i = 1:160
        [minDistance, index] = min(distanceArray(:,i));
        closestClass(:,i) = [F32(1,i); F32(2,i); index];
    end
    
    % Calculate new mean
    newMeans = zeros(2,10);
    for i = 1:10
        newCluster = find(closestClass(3,:) == i);
        newMeans(:,i) = [sum(closestClass(1,newCluster)); sum(closestClass(2,newCluster))] / numel(newCluster);
    end
end