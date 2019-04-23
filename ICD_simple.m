% Iterative reconstruction implemented using iterative coordinate descent.
% References several helper functions at bottom of document. 
%
% Scott Hsieh - January 2017

function recon = ICD_simple(ground_truth, params)

% One-line helper function: show the image enlarged 3x
displayImage = @(x) (imshow(imresize(x, 3, 'nearest'), params.displayWindow, 'Border', 'Tight'));

% One-line helper function that implements wrap-around
incrementWithWrap = @(x, delta) ( mod(x + delta - 1,params.imageSize) + 1);

% Generate noiseless raw data using our projector, by forward projecting each 
% individual pixel. Note that this is extremely ideal, because we've
% eliminated the possibility of any geometry mismatch; we are making the
% raw data perfectly in line with our projector.
rawDataNoNoise = zeros(params.nViews, params.nDets);
for xx = 1:params.imageSize
    for yy = 1:params.imageSize
        % This is the simplest possible pixel-driven forward projector,
        % like RADON but without subpixel division.
        rawDataNoNoise = rawDataNoNoise + forwardProjectSlow(ground_truth, [xx yy], params.nViews, params.nDets);
    end
end

% Determine the variance for each point in the sinogram. These lines don't
% matter so much, as they're just used to generate varianceSino as a tool
% to add noise into the sinogram
statisticalWeights = rawDataNoNoise * 0;
for kk = 1:params.nViews
    statisticalWeights(kk,:) = params.tcmFunction(kk);
end
statisticalWeights = statisticalWeights ./ mean(statisticalWeights(:));
varianceSino = 1 ./ statisticalWeights * params.noiseLevel^2;

% Add Gaussian noise into the raw data, following the variance sinogram
rawData = rawDataNoNoise + randn(size(rawDataNoNoise)) .* sqrt(varianceSino);

% The explicit calculation of the cost function isn't needed for the
% functioning of the algorithm, but it's a good debugging tool. The cost
% function should monotonically decrease as we iterate in ICD until we
% converge. 
nowCost = costFunc_huberAndWeights(rawData, rawData, ground_truth, params.huberCutoff, params.regularizerStrength , statisticalWeights);
disp(['cost of noiseless ground truth data:' num2str(nowCost, 4)]);

% for a starting guess -- use Matlab's IRADON
directRecon = iradon(rawData', 180/pi*angleSpace(params.nViews), 'linear', 'Hamming');
directRecon = directRecon * 1;
% possibly shrink it by one pixel because of convention mismatch
directRecon = directRecon(1:params.imageSize,1:params.imageSize);
changer = directRecon*0 + 1; % temporary variable we'll use later

% Take your initial guess (directRecon) and map it with the projector. So
% currentSino stores the current forward projection of the reconstruction
disp('Calculating current sinogram of starting guess ...');
currentSino = rawData * 0;
for xx = 1:params.imageSize
    for yy = 1:params.imageSize
        currentSino = currentSino + forwardProjectSlow(directRecon, [xx yy], params.nViews, params.nDets);
    end
end
% ... and current reconstruction is the current reconstruction
currentRecon = directRecon;

% Loop over desired number of iterations
for masterItr = 1:params.nIterations
    % Get a random coordinate traversal order, store in the useOrder variable
    % The rows of useOrder contain all combinations of pixel coordinates [x, y]
    % but stored in a shuffled fashion
    [xx, yy] = meshgrid(1:params.imageSize, 1:params.imageSize);
    traversalOrder = randperm(params.imageSize*params.imageSize)';
    useOrder = [xx(traversalOrder), yy(traversalOrder)];

    % Calculate the current cost function, just for reference
    nowCost = costFunc_huberAndWeights(currentSino, rawData, currentRecon, params.huberCutoff, params.regularizerStrength, statisticalWeights);
    disp(['Beginning iteration ' int2str(masterItr) ' at cost ' num2str(nowCost, 8)]);

    for subitr = 1:(params.imageSize^2)
        % Each sub-iteration operates on one pixel in the loop, whose
        % coordinates are provided by useOrder(subitr,:).
        
        % Calculate the current mismatch in the sinogram. Everywhere the
        % voxel projects onto, we can calculate the contribution to the
        % cost function. This will produce a parabola. We can add up all
        % the parabolas for the total mismatch cost function, which will
        % still be a parabola.
        dSino = forwardProjectSlow(changer, useOrder(subitr,:), params.nViews, params.nDets);
        parabola_linearTerm = sum(sum((rawData - currentSino) .* statisticalWeights .* dSino * (-2)));
        parabola_quadraticTerm = sum((dSino(:) .^ 2) .* statisticalWeights(:));
                
        % Next, let's look at the smoothing or regularizer part of the cost
        % function. First get the pixel's coordinates:
        thisR = useOrder(subitr,1);
        thisC = useOrder(subitr,2);
        
        % Now iterate over the 8 neighbors and calculate the difference:
        neighborVals = zeros(8,1);
        ctr = 1;
        for dx = -1:1
            for dy = -1:1
                if (dx == 0) && (dy == 0)
                    continue;
                end
                % The 'wrapping' part is not physically meaningful but is a
                % simple (incorrect) way to avoid errors at the boundary
                neighborVals(ctr) = currentRecon(incrementWithWrap(thisR, dx), ...
                    incrementWithWrap(thisC, dy)) - currentRecon(thisR, thisC);
                ctr = ctr + 1;
            end
        end
        
        % Consider a group of 100 candidate values between -0.5 and 0.5.
        % Evaluate the Huber functions and sum them for each of the
        % neighbors
        dAdd = [linspace(-0.5, 0.5, params.search1D) 0]; % last entry is special, no change
        
%         % Slow version -- find the minimum 
%        dHuber = dAdd * 0;
%         for kk = 1:numel(dHuber)
%             diff = abs(neighborVals - dAdd(kk));
%             temp = diff * 0;
%             temp(diff <= HUBER_CUTOFF) = diff(diff <= HUBER_CUTOFF) .^ 2;
%             temp(diff > HUBER_CUTOFF) = diff(diff > HUBER_CUTOFF) * 2 * HUBER_CUTOFF - HUBER_CUTOFF .^2;
%             dHuber(kk) = sum(temp(:));
%         end

        % Vectorized, faster version: calculate the Huber regularizer
        % penalty for the range of candidate values
        alternative = repmat(neighborVals, [1 numel(dAdd)]);
        diff = abs(bsxfun(@minus, alternative, dAdd));
        temp = diff * 0;
        temp(diff <= params.huberCutoff) = diff(diff <= params.huberCutoff) .^ 2;
        temp(diff > params.huberCutoff) = diff(diff > params.huberCutoff) * 2 * params.huberCutoff - params.huberCutoff .^2;
        dHuber = sum(temp);
        
        % Get a difference of Huber function values by subtracting the
        % current value. This line only makes a difference for debugging
        dHuber = dHuber - dHuber(end); 
        
        dCost = (parabola_quadraticTerm .* dAdd .^ 2 + ...
            parabola_linearTerm .* dAdd) + ...
            dHuber * params.regularizerStrength  * 2; % double it because it gets double counted in regularizer
        
        [reductionInCost, idx] = min(dCost);
        addAmount = dAdd(idx); % this is the amount of change we want to push in
        
        % Update our sinogram and reconstruction
        currentSino = currentSino + dSino * addAmount;
        currentRecon(useOrder(subitr,1),useOrder(subitr,2)) = ...
            currentRecon(useOrder(subitr,1),useOrder(subitr,2)) + addAmount;
    end
    figure(2);
    displayImage(currentRecon);
    getframe;
end

% Annotate, display, and package results into the 'recon' struct
disp('Showing starting guess | current reconstruction');
displayImage([directRecon currentRecon]);
text(20,25,'FBP','Color', 'w', 'FontSize', 20);
text(6*params.imageSize-80,25,'ICD','Color', 'w', 'FontSize', 20);

recon.iterative = currentRecon;
recon.fbp = directRecon;

end

function sino = forwardProjectSlow(input, pixelLoc, nViews, nDets)
    % Forward project a given pixel in the sinogram.
    % Extract coordinates of the pixel
    dy = pixelLoc(1) - (size(input,1) + 1)/2;
    dx = pixelLoc(2) - (size(input,2) + 1)/2;
    
    % Project this onto the detector.
    viewSpace = angleSpace(nViews);
    
    sino = zeros(nViews, nDets);
    
    for vv = 1:nViews
        detIdx = dx * cos(viewSpace(vv)) - dy * sin(viewSpace(vv)) + (nDets+1)/2;
        sino(vv, round(detIdx)) = input(pixelLoc(1), pixelLoc(2));
    end
end

% Helper function to generate a set of angles uniformly from [0, pi)
function angles = angleSpace(N)
    angles = pi/180*linspace(0,180-180/N,N);
end

% This computes the global cost function across the entire image.
% This is only needed as a reference or for debugging; it's not needed for
% the main program logic
function cost = costFunc_huberAndWeights(fpSino, rawData, recon, huberCutoff, weighting, statistics)
    costData = sum( statistics(:) .* (rawData(:) - fpSino(:)).^2);
    costReg = 0;
    for dx = -1:1
        for dy = -1:1
            if (dx == 0) && (dy == 0) % save some time
                continue;
            end
            diff = recon - circshift(recon, [dx dy]);
            diff = abs(diff);
            temp = diff * 0;
            temp(diff <= huberCutoff) = diff(diff <= huberCutoff) .^ 2;
            temp(diff > huberCutoff) = diff(diff > huberCutoff) * 2 * huberCutoff - huberCutoff .^2;
            costReg = costReg + sum(temp(:));
        end
    end

    cost = costData + weighting * costReg;
end