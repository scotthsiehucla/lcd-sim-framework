% SIMULATIONS OF A RAPID DETECTABILITY ANALYSIS USING A MODEL OBSERVER
% Scott Hsieh - UCLA
% April 2019
%
% This code provides simulations to test the feasibility of a rapid
% detectability analysis. Several intermediate outputs are shown, but in
% the current code, they are not archived (a breakpoint is needed to
% examine intermediate results).

clear all
close all

% accounting for the figures we need at the end
figureHolder = [];
figureCtr = 1;

for runType = {'Asymmetric', 'Uniform' }

    % Set the basic parameters
    params.imageSize = 129;      % side length of the image
    params.nViews = 101;         % number of views
    params.nDets = 185;          % number of detector channels

    params.displayWindow = [-1 1];  % Affects display only
    params.nIterations = 4;         % Number of iterations to use

    % Affects regularization of ICD (tuned empirically)
    params.huberCutoff = 3;
    params.regularizerStrength = 30;

    % Affects noise statistics: both globally and in a view-dependent way
    params.noiseLevel = 12.5 * 1;
    
    % Run in twice, and change the nature of the tube current modulation.
    % This alters the statistics.
    if strcmp(runType, 'Asymmetric')
        params.tcmFunction = @(x) (sin(x/params.nViews * 2 * pi) + 1.2);
    else
        params.tcmFunction = @(x) (sin(x/params.nViews * 2 * pi)*0 + 1);
    end

    % seed the random number generation consistently to get reproducible
    % results
    rng(0); 

    % Create the target noiseless image.
    % Populate it with 50 lesion entries, and throw out any lesions that
    % are too close to another lesion.
    basicSignal = zeros(params.imageSize);
    truthX = []; truthY = [];
    listList = rand(50,2) .* params.imageSize;
    xList = listList(:,1);
    yList = listList(:,2);
    [xx, yy] = meshgrid(1:params.imageSize, 1:params.imageSize);
    for kk = 1:numel(xList)
        dd = sqrt((xList - xList(kk)) .^ 2  + (yList - yList(kk)) .^ 2);
        dd = sort(dd);
        if (dd(2) < 8)
            disp([kk dd(1:4)'])
            continue;
        end
        truthX = [truthX; xList(kk)];
        truthY = [truthY; yList(kk)];
        rr = sqrt( (xx - xList(kk)) .^ 2  + (yy - yList(kk)) .^ 2);
        basicSignal(rr < 3) = 0.5;
    end
    imshow(imresize(basicSignal, 2), [], 'border', 'tight');
    text(20,20,'Noiseless', 'FontSize', 20, 'Color', 'w', 'FontWeight', 'bold');

    % This line of code is repeated throughout the script and only serves
    % to archive a figure for display later.
    f = getframe; figureHolder{figureCtr} = f.cdata; figureCtr = figureCtr + 1;
    
    % Do the iterative recon.
    params.search1D = 100;
    recon = ICD_simple(basicSignal, params);

    % Show the images
    imshow(imresize(recon.fbp, 2), [], 'border', 'tight');
    text(20,20,'FBP', 'FontSize', 20, 'Color', 'w', 'FontWeight', 'bold');
    text(20,45,runType, 'FontSize', 20, 'Color', 'w', 'FontWeight', 'bold');
    
    % Save this figure for later
    f = getframe; figureHolder{figureCtr} = f.cdata; figureCtr = figureCtr + 1;

    imshow(imresize(recon.iterative, 2), [], 'border', 'tight');
    text(20,20,'Iterative', 'FontSize', 20, 'Color', 'w', 'FontWeight', 'bold')
    text(20,45,runType, 'FontSize', 20, 'Color', 'w', 'FontWeight', 'bold');

    % Save this figure for later
    f = getframe; figureHolder{figureCtr} = f.cdata; figureCtr = figureCtr + 1;
    
    % Convolve and do the NPW detection process
    % Create a circle
    [xx, yy] = meshgrid(1:16,1:16);
    rr = sqrt( (xx - 8.5) .^ 2  + (yy - 8.5) .^ 2);
    maskIt = xx * 0;
    maskOut = xx * 0;
    maskIt(rr < 3) = 1;
    maskIt = maskIt ./ sum(maskIt(:));
    for analyze_type = 1:2
        if (analyze_type == 1)
           orig = recon.fbp;
        else
           orig = recon.iterative;
        end
        trial = imfilter(orig, maskIt);
        imshow(trial, []);
        isMax = trial .^ 0;
        for dx = -6:6
            for dy = -6:6
                isMax = isMax & (trial >= circshift(trial, [dx dy]));
            end
        end
        imshow(isMax, []);

        frac_found = [];
        frac_fp = [];

        true_locs = orig * 0;
        for kk = 1:numel(truthX)
            true_locs(round(truthY(kk)), round(truthX(kk))) = 1;
        end
        ch = zeros(13,13); ch(7,[1:3 11:13]) = 1; ch([1:3 11:13],7) = 1;
        boxbox = zeros(13,13); boxbox([1 end],:) = 1; boxbox(:, [1 end]) = 1;
        true_locs = imfilter(true_locs, ch);

        for thresh = 0.1:0.01:0.8 % tune the threshold range for NPW
            rawList = isMax & trial > thresh;

            showMe = orig;
            showMe = showMe - min(showMe(:));
            showMe = showMe ./ max(showMe(:));
            showMe(:,:,2) = showMe(:,:,1);
            showMe(:,:,3) = showMe(:,:,1);
            showMe(:,:,1) = showMe(:,:,1) + imfilter(rawList, boxbox)*0.4;
            showMe(:,:,2) = showMe(:,:,2) + imfilter(rawList, boxbox)*0.4 + true_locs * 0.4;
            showMe(:,:,3) = showMe(:,:,3) + true_locs * 0.4;
            imshow(imresize(showMe, 2), [], 'border', 'tight')

            [xf, yf] = find(rawList);
            trues = truthX * 0;
            falseCount = 0;
            searchRad = 3;
            distIt = @(x, y, xList, yList) ( (sqrt( (x - xList).^2 + (y - yList) .^ 2) ) );
            for kk = 1:numel(xf)
                dd = distIt(yf(kk), xf(kk), truthX, truthY); % need to switch row/col here
                [dd, id] = sort(dd);
                if dd(1) < searchRad
                    if trues(id(1)) == 0
                        trues(id(1)) = trues(id(1)) + 1;
                    else
                        falseCount = falseCount + 1; % double counted
                    end
                else
                    falseCount = falseCount + 1;
                end
            end
            disp(['Fraction trues found: ' num2str(sum(trues)/numel(truthX))])
            disp(['False positives: ' num2str(falseCount)])
            disp(['False positive fraction: ' num2str(falseCount / sum(trues))])
            temp = sum(trues)/numel(truthX);
            if isnan(temp)
                temp = 0;
            end
            frac_found = [frac_found, temp];
            frac_fp = [frac_fp, falseCount ];
        end
        plot(frac_fp, frac_found, '.-');
        xlabel('False positives');
        ylabel('Sensitivity');
        axis([0 10 0 1]);
        if (analyze_type == 1)
            fbp_x = frac_fp;
            fbp_y = frac_found;
        else
            ir_x = frac_fp;
            ir_y = frac_found;
        end
    end
    
    % Display the final results
    close all;
    imshow(ones(128,128));
    plot(fbp_x, fbp_y);
    hold all;
    plot(ir_x, ir_y);
    hold off
    axis([0 10 0 1]);
    xlabel('False positives'); ylabel('Sensitivity'); title('Free response ROC');
    legend('FBP', 'Iterative');
    % Store data for analysis later
    f = getframe; 
    figureHolder{figureCtr} = [];
    figureHolder{figureCtr}.fig = f.cdata;
    figureHolder{figureCtr}.dataFBPx = fbp_x;
    figureHolder{figureCtr}.dataFBPy = fbp_y;
    figureHolder{figureCtr}.dataIRx = ir_x;
    figureHolder{figureCtr}.dataIRy = ir_y;
    figureCtr = figureCtr + 1;
end

% Display the results in two summary figures.

figure(1);
imshow([figureHolder{1}, figureHolder{2}, figureHolder{3}, ...
    figureHolder{6}, figureHolder{7}], 'border', 'tight');

figure(2);
plot(figureHolder{4}.dataFBPx,figureHolder{4}.dataFBPy, 'k-', 'LineWidth', 2);
hold all;
plot(figureHolder{4}.dataIRx,figureHolder{4}.dataIRy, 'k-.', 'LineWidth', 2);
plot(figureHolder{8}.dataFBPx,figureHolder{4}.dataFBPy, 'b-', 'LineWidth', 2);
plot(figureHolder{8}.dataIRx,figureHolder{4}.dataIRy, 'b-.', 'LineWidth', 2);
hold off
legend('FBP - Asymmetric', 'IR - Asymmetric', 'FBP - Uniform', 'IR - Uniform');
axis([0 10 0 1]);
xlabel('False positives'); ylabel('Sensitivity'); title('Free response ROC');
title('Detectability of IR and FBP')