clear
clc

set(groot,'defaultAxesTickLabelInterpreter', 'latex');
set(groot,'defaultLegendInterpreter', 'latex');
set(groot,'defaultTextInterpreter', 'latex');


n_samp = 4000;
mode = "A03";
filenamepart =  "_4000";
%%
switch mode
    case "A01"
        ny = 64;
    case "A03" 
        ny = 64;
    case "B04"
        ny = 48;
    case "C04"
        ny = 32;
end



H = 1.75;
q = 5;
% q = 1; quadrant Q1 u+ v+
% q = 2; quadrant Q2 u- v+
% q = 3; quadrant Q3 u- v-
% q = 4; quadrant Q4 u+ v-
% q = 5; All quadrants Q1 Q2 Q3 Q4

limit = 0.3:0.05:1;
%limit = 0.3:0.1:0.4;




cum_num_str_total  = [0,0];
cum_vol_str_total  = [0,0];
cum_voly_str_total = [0,0];
cum_num_str_match  = zeros(numel(limit),2);
cum_vol_str_match  = zeros(numel(limit),2);
cum_voly_str_match = zeros(numel(limit),2);
Xt.ymin = [];
Xt.ymax = [];
Xt.Pvol = [];
Xt.Xvol = [];
Xp.ymin = [];
Xp.ymax = [];
Xp.Pvol = [];
Xp.Xvol = [];

tic

for file_idx = 1:n_samp/500

load(char("GANS_matlab_matrices\" + "matlabmatrix" + mode + filenamepart + "_" + file_idx + ".mat"))
%load(char("matlabmatrixprueba"+mode+".mat"))
%   # data = {'x': X, 'y': Y, 'z': Z, 'target': y_target, 'predic': y_predic}
%   # savemat('matlabmatrix.mat', data)
%   ### target, predic - 5D matrix - [sample, x[1:nx], y[1:ny], z[1:nz], V[u,v,w]]
%   ### OR
%   ### target, predic - 4D matrix - [x[1:nx], y[1:ny], z[1:nz], V[u,v,w]]

target(:,:,1,:,:) = 0;
predic(:,:,1,:,:) = 0;

nx = length(x);
nz = length(z);

y = y(1:ny);

target = target(:,:,1:ny,:,:);
predic = predic(:,:,1:ny,:,:);

snaps = size(target,1);

rms_ut = sqrt( sum( (reshape(permute(target(:,:,:,:,1),[1,2,4,3]),[snaps*nx*nz,ny])).^2) / (snaps*nx*nz));
rms_vt = sqrt( sum( (reshape(permute(target(:,:,:,:,2),[1,2,4,3]),[snaps*nx*nz,ny])).^2) / (snaps*nx*nz));
rms_up = sqrt( sum( (reshape(permute(predic(:,:,:,:,1),[1,2,4,3]),[snaps*nx*nz,ny])).^2) / (snaps*nx*nz));
rms_vp = sqrt( sum( (reshape(permute(predic(:,:,:,:,2),[1,2,4,3]),[snaps*nx*nz,ny])).^2) / (snaps*nx*nz));

for snap = 1:size(target,1)

    Qt=zeros(nx,ny,nz,'logical');  Qp=zeros(nx,ny,nz,'logical');

    for i = 1:length(x)
        for j = 1:length(y)
            for k = 1:length(z)
                
                if q == 1  % quadrant Q1 u+ v+
                    if target(snap,i,j,k,1) > 0 && target(snap,i,j,k,2) > 0
                        if abs(target(snap,i,j,k,1) .* target(snap,i,j,k,2)) > H * rms_ut(j) * rms_vt(j)
                            Qt(i,j,k) = 1;
                        end
                    end
                    if predic(snap,i,j,k,1) > 0 && predic(snap,i,j,k,2) > 0
                        if abs(predic(snap,i,j,k,1) .* predic(snap,i,j,k,2)) > H * rms_up(j) * rms_vp(j)
                            Qp(i,j,k) = 1;
                        end
                    end
                end
                
                
                
                if q == 2  % quadrant Q2 u- v+
                    if target(snap,i,j,k,1) < 0 && target(snap,i,j,k,2) > 0
                        if abs(target(snap,i,j,k,1) .* target(snap,i,j,k,2)) > H * rms_ut(j) * rms_vt(j)
                            Qt(i,j,k) = 1;
                        end
                    end
                    if predic(snap,i,j,k,1) < 0 && predic(snap,i,j,k,2) > 0
                        if abs(predic(snap,i,j,k,1) .* predic(snap,i,j,k,2)) > H * rms_up(j) * rms_vp(j)
                            Qp(i,j,k) = 1;
                        end
                    end
                end
                
                
                
                if q == 3  % quadrant Q3 u- v-
                    if target(snap,i,j,k,1) < 0 && target(snap,i,j,k,2) < 0
                        if abs(target(snap,i,j,k,1) .* target(snap,i,j,k,2)) > H * rms_ut(j) * rms_vt(j)
                            Qt(i,j,k) = 1;
                        end
                    end
                    if predic(snap,i,j,k,1) < 0 && predic(snap,i,j,k,2) < 0
                        if abs(predic(snap,i,j,k,1) .* predic(snap,i,j,k,2)) > H * rms_up(j) * rms_vp(j)
                            Qp(i,j,k) = 1;
                        end
                    end
                end
                
                
                
                if q == 4  % quadrant Q4 u+ v-
                    if target(snap,i,j,k,1) > 0 && target(snap,i,j,k,2) < 0
                        if abs(target(snap,i,j,k,1) .* target(snap,i,j,k,2)) > H * rms_ut(j) * rms_vt(j)
                            Qt(i,j,k) = 1;
                        end
                    end
                    if predic(snap,i,j,k,1) > 0 && predic(snap,i,j,k,2) < 0
                        if abs(predic(snap,i,j,k,1) .* predic(snap,i,j,k,2)) > H * rms_up(j) * rms_vp(j)%rms_up(j) * rms_vp(j)
                            Qp(i,j,k) = 1;
                        end
                    end
                end
                
                
                
                if q == 5  % All quadrants Q1 Q2 Q3 Q4
                        if abs(target(snap,i,j,k,1) .* target(snap,i,j,k,2)) > H * rms_ut(j) * rms_vt(j)
                            Qt(i,j,k) = 1;
                        end
                        if abs(predic(snap,i,j,k,1) .* predic(snap,i,j,k,2)) > H * rms_up(j) * rms_vp(j)
                            Qp(i,j,k) = 1;
                        end
                end
                
                
            end
        end
    end
    
    
    
    
    
    
    
    [aux_t, CC_t] = connectivity(Qt,x(2),y,z(2),nx,ny,nz);
    if snap == 1
        stats_t = aux_t;
    elseif  size(aux_t,1) > 0
        stats_t = [stats_t; aux_t]; 
    end
    
    [aux_p, CC_p] = connectivity(Qp,x(2),y,z(2),nx,ny,nz);
    if snap == 1 
        stats_p = aux_p;
    elseif size(aux_p,1) > 0 && size(stats_p,1) > 1
        stats_p = [stats_p; aux_p];
    elseif snap > 1 && size(stats_p,1) == 0
        stats_p = aux_p;
    end
   
    
     
    

    dy=zeros(1,numel(y));
    dy(1) = (y(1)+y(2))/2;
    dy(2:numel(dy)-1) = ( y(3:numel(dy)) - y(1:numel(dy)-2)) /2;
    dy(numel(dy)) = ( y(numel(dy)) - y(numel(dy)-1));
    
    [CC_mt, CC_mp, CC_ut, CC_up, Xtt, Xpp] = match_ymin_ymax(CC_t, CC_p, nx, ny, nz, dy, y);
    
%         Xt = [Xt Xtt];
%         Xp = [Xp Xpp];

    Xt.ymin = [Xt.ymin Xtt.ymin];
    Xt.ymax = [Xt.ymax Xtt.ymax];
    Xt.Pvol = [Xt.Pvol Xtt.Tvol];
    Xt.Xvol = [Xt.Xvol Xtt.Xvol];
    Xp.ymin = [Xp.ymin Xpp.ymin];
    Xp.ymax = [Xp.ymax Xpp.ymax];
    Xp.Pvol = [Xp.Pvol Xpp.Pvol];
    Xp.Xvol = [Xp.Xvol Xpp.Xvol];

       
   

end
file_idx
end


%     switch mode
%         case "A01"
%             step = 0.05;
%         case "A03"
%             step = 0.05;
%         case "B04"
%             step = 0.05;
%         case "C04"
%             step = 0.05;
%     end
%     box = {0:step:step*ceil(y(numel(y))/step), 0:step:step*ceil(y(numel(y))/step)};
switch mode
    case "A01"
        box = {[0,0.0993,0.1433,0.2067,0.3271,0.5175,0.7194,1] , [0,0.0993,0.1433,0.2067,0.3271,0.5175,0.7194,1]};
    case "A03"
        box = {[0,0.0993,0.1433,0.2067,0.3271,0.5175,0.7194,1] , [0,0.0993,0.1433,0.2067,0.3271,0.5175,0.7194,1]};
    case "B04"
        box = {[0,0.0993,0.1433,0.2067,0.3271,0.5175] , [0,0.0993,0.1433,0.2067,0.3271,0.5175]};
    case "C04"
        box = {[0,0.0993,0.1433,0.2067] , [0,0.0993,0.1433,0.2067]};
end

for time=1:3
    switch time
        case 1
            [H, Hcount_t] = histo_ymin_ymax(box,Xt); H_t = H;
        case 2
            [H, Hcount_p] = histo_ymin_ymax(box,Xp); H_p = H;
        case 3
            H = H_t .* H_p;
    end
    
%     s = sum(H,'all');
%         for i = 1:size(H,2)
%             H(i,i) = 2 * H(i,i);
%         end
    H = H';
%     H = H / s;
    
    figure
    
    % contourf(log10(H2),10)
    imagesc((H))
    cmap = flip(hot) ; 
    colormap(cmap)

    set(gca, 'YDir', 'normal');
%         colorbar
%         colorbar('Ticks',[0 0.2 0.4 0.6 0.8 1])
    caxis([0, 1])

    hold on
    pgon = polyshape([0 size(H,2)+0.5 size(H,2)+0.5],[0 0 size(H,2)+0.5]);
    plot(pgon, 'facecolor','white','EdgeColor',[0.75 0.75 0.75],'FaceAlpha',1)

    xticks(linspace(0.5,size(H,2)+0.5,size(box{1,1},2)))
    xticklabels(newlabels(box{1,2}))
    yticks(linspace(0.5,size(H,2)+0.5,size(box{1,2},2)))
    yticklabels(newlabels(box{1,1}))
    
    xlabel('$y_{\rm min}/h$','interpreter','latex')
    ylabel('$y_{\rm max}/h$','interpreter','latex')
    axis equal
    xlim([1-0.5, size(H,2)+0.5])
    ylim([1-0.5, size(H,1)+0.5])
    grid on
    set(gca,'fontsize',15)
    
    switch time
        case 1
            title('$X_t$','interpreter','latex')
        case 2
            title('$X_p$','interpreter','latex')
        case 3
            title('$X_t X_p$','interpreter','latex')
    end

    % HISTOGRAM
    threshold = 0.95;
    if time == 1
        Hcount_t_hist = hist_counter (Hcount_t, threshold, box{1,1}); aux = Hcount_t_hist;
    elseif time ==2
        Hcount_p_hist = hist_counter (Hcount_p, threshold, box{1,1}); aux = Hcount_p_hist;
    end

    if time < 3
        for i = 1:size(H,1)
            for j = 1:size(H,2)
                if aux(i,j) == 1
                    scatter(i,j,50,'filled','MarkerFaceColor','k')
                end
            end
        end
    end
end
  




%%



% plotbinaryvolume(Qt,x,y,z,nz)
% plotbinaryvolume(Qt,x,y,z,nz)
% plotbinaryvolume(Qt,x,y,z,nz)
% figure
% plotbinaryvolume(Qt,x,y,z,nz)

 % 
    % 
    % hold on
    % 
    % for str = 1:size(stats,1)
    %     plot3(stats.Centroid(str,2),z(nz)-stats.Centroid(str,3),stats.Centroid(str,1),'r*')
    % end



mean_target = statistics(stats_t);
mean_predic = statistics(stats_p);

volume_target = mean_target.meanVolume(size(mean_target,1)) * size(mean_target,1) / snaps;
volume_predic = mean_predic.meanVolume(size(mean_predic,1)) * size(mean_predic,1) / snaps;
volume_percentage = volume_predic / volume_target;
V_target_perc = volume_target / (pi^2/2*y(numel(y)));
V_predic_perc = volume_predic / (pi^2/2*y(numel(y)));

stats_ta = stats_t;     stats_td = stats_t;

stats_pa = stats_p;     stats_pd = stats_p;

for ee = (size(stats_t,1):-1:1)
    if stats_ta.Attached(ee)
        stats_td(ee,:) = [];
    else
        stats_ta(ee,:) = [];
    end
end

for ee = (size(stats_p,1):-1:1)
    if stats_pa.Attached(ee)
        stats_pd(ee,:) = [];
    else
        stats_pa(ee,:) = [];
    end
end



toc


%%
switch mode
    case "A01"
        y_len = 1;
    case "A03"
        y_len = 1;
    case "B04"
        y_len = (y(48)+y(49))/2;
    case "C04"
        y_len = (y(32)+y(33))/2;
end
vol_total = pi * pi/2 * y_len;
fprintf('\nTotal volume %4.4f \n', vol_total)

fprintf('\nTARGET')
vol_Xt = sum(Xt.Pvol/n_samp *pi/64 *pi/2/64);
fprintf('\nTarget volume %4.4f %2.2f%% \n', vol_Xt, vol_Xt/vol_total*100)
tot = sum(sum(Hcount_t))-2.1; att = sum(Hcount_t(1,:));
fprintf('\nTotal Structures %5.0f --> %3.3f \nAttached structures %5.0f %2.2f%% \n', tot , tot/n_samp, att, 100*att/tot)

fprintf('\nPREDICT')
vol_Xp = sum(Xp.Pvol/n_samp *pi/64 *pi/2/64);
fprintf('\nPredict volume %4.4f %2.2f%% \n', vol_Xp, vol_Xp/vol_total*100)
tot = sum(sum(Hcount_p))-2.1; att = sum(Hcount_p(1,:));
fprintf('\nTotal Structures %5.0f --> %3.3f  \nAttached structures %5.0f %2.2f%% \n', tot , tot/n_samp, att, 100*att/tot)

%%





function plotbinaryvolume(arg,x,y,z,nz,tit)
[X,Z,Y] = meshgrid(x,z,y);

arg = permute(arg,[3,1,2]);
arg = flipud(arg);
isosurface(X,Z,Y,arg,0.9999,Y)
xlabel('$x/h$','interpreter','latex');
ylabel('$z/h$','interpreter','latex');
zlabel('$y/h$','interpreter','latex');
yticks([z(nz)-1.5 z(nz)-1 z(nz)-0.5 z(nz) ])
yticklabels({'1.5','1','0.5','0'})
title(tit)
grid on
axis equal
end

function plotbinaryvolume9(arg,x,y,z,nz,tit)
x = [x-x(2)*numel(x), x, x+x(2)*numel(x)];
z = [z-z(2)*numel(z), z, z+z(2)*numel(z)];
[X,Z,Y] = meshgrid(x,z,y);


plot3([x(numel(x)/3+1) x(2/3*numel(x))], [z(numel(z)/3+1) z(2/3*numel(z))], [y(1) y(numel(y))],'ow')
hold on

arg = permute(arg,[3,1,2]);
arg = flipud(arg);
isosurface(X,Z,Y,arg,0.9999,Y)
xlabel('$x/h$','interpreter','latex');
ylabel('$z/h$','interpreter','latex');
zlabel('$y/h$','interpreter','latex');
% yticks([z(2*nz)-1.5 z(2*nz)-1 z(2*nz)-0.5 z(2*nz) ])
% yticklabels({'1.5','1','0.5','0'})
title(tit)

xlim([x(1), x(numel(x))])
ylim([z(1), z(numel(z))])
zlim([y(1), y(numel(y))])
grid on
axis equal
end

function [stats, CC] = connectivity(arg,dx,y,dz,nx,ny,nz)

X = arg;
arg(:,:,nz+1:2*nz) = X;
arg(:,:,2*nz+1:3*nz) = X;
arg = [arg;arg;arg];

CC = bwconncomp(arg);
L = labelmatrix(CC);
stats = regionprops3(CC,'Centroid','Volume','BoundingBox');  %,'all'         % 'Centroid','Volume','BoundingBox'


dy(1) = (y(2)+y(1)) /2;
dy(2:numel(y)-1) = (y(3:numel(y)) - y(1:numel(y)-2)) /2;
dy(numel(y)) = y(numel(y)) - y(numel(y)-1);


for ii = (size(stats,1):-1:1)
    if stats.Centroid(ii,2) <= nx || stats.Centroid(ii,2) > 2*nx || stats.Centroid(ii,3) <= nz || stats.Centroid(ii,3) > 2*nz
        stats(ii,:) = [];
        CC.PixelIdxList(ii) = [];
    end
end

for ii = (size(stats,1):-1:1)
    if stats.Volume(ii) == 1 || stats.BoundingBox(ii,5) == 1 || stats.Centroid(ii,1) == 1
        stats(ii,:) = [];
        CC.PixelIdxList(ii) = [];
    end
end

if size(stats,1) > 0
    stats.Centroid(:,2) = stats.Centroid(:,2) - nx;
    stats.Centroid(:,3) = stats.Centroid(:,3) - nz;
end

for e = 1:size(stats,1)
    stats.Centroid(e,2) = (stats.Centroid(e,2) -1) * dx;
    stats.Centroid(e,3) = (stats.Centroid(e,3) -1) * dz;
    
    y_array = ceil(CC.PixelIdxList{1,e} /(3*nx));
    y_array = y_array - floor(y_array./ny - 1/(2*ny)) .* ny;

    stats.Centroid(e,1) = sum(y(y_array)) / size(CC.PixelIdxList{1,e},1);

    stats.Volume(e) = dx * dz * sum(dy(y_array));

    stats.ymin(e) = y(min(y_array));
    stats.ymax(e) = y(max(y_array));

    stats.Lx(e) = (stats.BoundingBox(e,5) -1) * dx;
%     stats.Ly(e) = stats.BoundingBox(e,4);
    stats.Lz(e) = (stats.BoundingBox(e,6) -1) * dz;

    if min(y_array) < 23      % y vector  ---->    y(23) = 0.1 -----> 0.1*200 = 20^+
        stats.Attached(e) = 1;
    else
        stats.Attached(e) = 0;
    end

end

for ii = (size(stats,1):-1:1)
    if stats.Volume(ii) < 1e-5 || stats.BoundingBox(ii,4)==1 || stats.BoundingBox(ii,5)==1 ||stats.BoundingBox(ii,6)==1 
        stats(ii,:) = [];
        CC.PixelIdxList(ii) = [];
    end
end

for ii = (size(stats,1):-1:1)
    if stats.Lx(ii) == 0 || stats.Lz(ii) == 0
        stats(ii,:) = [];
        CC.PixelIdxList(ii) = [];
    end
end

% % CONDITION
% for ii = (size(stats,1):-1:1)
%     if stats.Centroid(ii,1) < 0.1
%         stats(ii,:) = [];
%         CC.PixelIdxList(ii) = [];
%     end
% end


end






function X = statistics(stats)

X = table(mean(stats.Volume), mean(stats.Lx), mean(stats.Lz), std(stats.Volume),  std(stats.Lx),  std(stats.Lz),...
    'VariableNames', {'meanVolume', 'meanLx', 'meanLz', 'stdVolume', 'stdLx' ,'stdLz'});
    for ii = 2:size(stats,1)
        X(ii,:) = table(mean(stats.Volume(1:ii)), mean(stats.Lx(1:ii)), mean(stats.Lz(1:ii)), std(stats.Volume(1:ii)),  std(stats.Lx(1:ii)),  std(stats.Lz(1:ii)));
    end
end





function [match_t, match_p, unmatch_t, unmatch_p] = match(all_t, all_p, nx, ny, nz, limit)

match_t.PixelIdxList = [];   % structures matched
match_p.PixelIdxList = [];
unmatch_t.PixelIdxList = [];   % structures unmatched
unmatch_p.PixelIdxList = [];

mat_corr = zeros(size(all_t.PixelIdxList,2), size(all_p.PixelIdxList,2),'logical');

for tt = 1:size(all_t.PixelIdxList, 2)
    for pp = 1:size(all_p.PixelIdxList, 2)

        vol_t = size(all_t.PixelIdxList{1,tt},1);
        vol_p = size(all_p.PixelIdxList{1,pp},1);

        T = zeros(3*nx, ny, 3*nz,'logical');
        P = zeros(3*nx, ny, 3*nz,'logical');

        for i = 1:vol_t
            [x, y, z] = num2coord (all_t.PixelIdxList{1,tt}(i), nx, ny);
            T(x,y,z) = 1;
        end

        for i = 1:vol_p
            [x, y, z] = num2coord (all_p.PixelIdxList{1,pp}(i), nx, ny);
            P(x,y,z) = 1;
        end

        X = T .* P;

        if sum(sum(sum(X))) / min(vol_t,vol_p) > limit
            mat_corr(tt,pp) = 1;
        end

    end
end

for tt = 1:size(all_t.PixelIdxList, 2)
    if sum(mat_corr(tt,:)) >= 1
        match_t.PixelIdxList{1,size(match_t.PixelIdxList,2)+1} = all_t.PixelIdxList{1,tt};
    else
        unmatch_t.PixelIdxList{1,size(unmatch_t.PixelIdxList,2)+1} = all_t.PixelIdxList{1,tt};
    end
end
for pp = 1:size(all_p.PixelIdxList, 2)
    if sum(mat_corr(:,pp)) >= 1
        match_p.PixelIdxList{1,size(match_p.PixelIdxList,2)+1} = all_p.PixelIdxList{1,pp};
    else
        unmatch_p.PixelIdxList{1,size(unmatch_p.PixelIdxList,2)+1} = all_p.PixelIdxList{1,pp};
    end
end

end












function [match_t, match_p, unmatch_t, unmatch_p, Xt, Xp] = match_ymin_ymax(all_t, all_p, nx, ny, nz, dy, yvect)

match_t.PixelIdxList = [];   % structures matched
match_p.PixelIdxList = [];
unmatch_t.PixelIdxList = [];   % structures unmatched
unmatch_p.PixelIdxList = [];

mat_corr = zeros(size(all_t.PixelIdxList,2), size(all_p.PixelIdxList,2));

Xt.ymin = zeros(1,size(all_t.PixelIdxList,2));
Xt.ymax = zeros(1,size(all_t.PixelIdxList,2));
Xt.Tvol = zeros(1,size(all_t.PixelIdxList,2));
Xt.Xvol = zeros(1,size(all_t.PixelIdxList,2));
Xp.ymin = zeros(1,size(all_p.PixelIdxList,2));
Xp.ymax = zeros(1,size(all_p.PixelIdxList,2));
Xp.Pvol = zeros(1,size(all_p.PixelIdxList,2));
Xp.Xvol = zeros(1,size(all_p.PixelIdxList,2));

for tt = 1:size(all_t.PixelIdxList, 2)

    size_t = size(all_t.PixelIdxList{1,tt},1);
    T = zeros(3*nx, ny, 3*nz,'logical'); % ones where there is a structure, zeros otherwise

    for i = 1:size_t    % ones where there is a structure, zeros otherwise
        [x, y, z] = num2coord (all_t.PixelIdxList{1,tt}(i), nx, ny);
        T(x,y,z) = 1;
    end

    vols = repmat(dy,[3*nx 1 3*nz]);

    Tvol = sum(sum(sum(T .* vols)));   % first bracket (volume where there is a structure / overlap, zeros otherwise)
                                       % sum of sturcture volume

    Xt.ymin(1, tt) = yvect(findymin(T));
    Xt.ymax(1, tt) = yvect(findymax(T));
    Xt.Tvol(1, tt) = Tvol;

    for pp = 1:size(all_p.PixelIdxList, 2)

        size_p = size(all_p.PixelIdxList{1,pp},1);
        P = zeros(3*nx, ny, 3*nz,'logical'); % ones where there is a structure, zeros otherwise

        for i = 1:size_p    % ones where there is a structure, zeros otherwise
            [x, y, z] = num2coord (all_p.PixelIdxList{1,pp}(i), nx, ny);
            P(x,y,z) = 1;
        end

        X = T .* P;         % ones where there is overlap, zeros otherwise
         
        Pvol = sum(sum(sum(P .* vols)));        % first bracket (volume where there is a structure / overlap, zeros otherwise)
        Xvol = sum(sum(sum(X .* vols)));        % sum of sturcture volume / overlap volume
        
        if tt == 1
            Xp.ymin(1, pp) = yvect(findymin(P));
            Xp.ymax(1, pp) = yvect(findymax(P));
            Xp.Pvol(1, pp) = Pvol;
        end

        if Xvol > 0 
            Xt.Xvol(1, tt) = Xt.Xvol(1, tt) + Xvol;
            Xp.Xvol(1, pp) = Xp.Xvol(1, pp) + Xvol;
        end
    end
end

for tt = 1:size(all_t.PixelIdxList, 2)
    if sum(mat_corr(tt,:)) >= 1
        match_t.PixelIdxList{1,size(match_t.PixelIdxList,2)+1} = all_t.PixelIdxList{1,tt};
    else
        unmatch_t.PixelIdxList{1,size(unmatch_t.PixelIdxList,2)+1} = all_t.PixelIdxList{1,tt};
    end
end
for pp = 1:size(all_p.PixelIdxList, 2)
    if sum(mat_corr(:,pp)) >= 1
        match_p.PixelIdxList{1,size(match_p.PixelIdxList,2)+1} = all_p.PixelIdxList{1,pp};
    else
        unmatch_p.PixelIdxList{1,size(unmatch_p.PixelIdxList,2)+1} = all_p.PixelIdxList{1,pp};
    end
end

end







function [x, y, z] = num2coord(a, nx, ny)
z = ceil(a / (3*nx*ny));
y = ceil(a / (3*nx)) - (z-1) * ny;
x = a - (y-1)*3*nx - (z-1)*3*nx*ny;
end

function A = str2num(CC, nx, ny, nz)
A = zeros(3*nx, ny, 3*nz, 'logical');
for i = 1:size(CC.PixelIdxList,2)
    for j = 1:size(CC.PixelIdxList{1,i},1)
        [x, y, z] = num2coord (CC.PixelIdxList{1,i}(j), nx, ny);
        A(x,y,z) = 1;
    end
end
end



function ymin = findymin (Y)
ymin=1;
while max(max(Y(:,ymin,:))) == 0
    ymin = ymin + 1;
end
end

function ymax = findymax (Y)
ymax=size(Y,2);
while max(max(Y(:,ymax,:))) == 0
    ymax = ymax - 1;
end
end





function [H, Hcount] = histo_ymin_ymax(box,X)
n = size(X.ymin,2);
H = zeros(size(box{1,1},2)-1);
Hcount = zeros(size(box{1,1},2)-1);

ymins = cell2mat(box(1,1));
ymaxs = cell2mat(box(1,2));

for nn=1:n
    i=1;
    while X.ymin(nn) > ymins(i+1)
        i = i+1;
    end
    j=i;
    while j<numel(ymaxs)-1 && X.ymax(nn) > ymaxs(j+1)
        j = j+1;
    end
    
    if X.Xvol(nn) > 0 
        H(i,j) = H(i,j) + X.Xvol(nn)/X.Pvol(nn);
        Hcount(i,j) = Hcount(i,j) + 1;
    end
end

for i = 1:size(H,1)
    for j = 1:size(H,2)
        if Hcount(i,j) == 0
            Hcount(i,j) = 0.1;
        end
    end
end
H = H./Hcount;

end






function X = newlabels(x)
    for i = 1:size(x,2)
        X{1,i} = sprintf('%0.2f',x(i));
    end
end


function X = hist_counter(H, threshold, box)
for i = 1:size(H,1)
    for j = 1:size(H,2)
        if H(i,j) < 1
            H(i,j) = 0;
        end
    end
end

% Area
A = zeros(numel(box)-1,numel(box)-1);
for i = 1:numel(box)-1
    for j = i:numel(box)-1
        A(i,j) = (box(i+1)-box(i)) * (box(j+1)-box(j));
    end
    A(i,i) = A(i,i) / 2;
end

H = H ./ A .* sum(sum(A));
H(isnan(H)) = 0;

Hmax = sort(reshape(H,1,[]),'descend');
Hmax_cum = cumsum(Hmax);
i = 1;
while Hmax_cum(i) < threshold*Hmax_cum(numel(Hmax_cum))
    i = i+1;
end

X = zeros(size(H));
for ind = 1:i
    found = 0; row = 1; col = 1;
    while found == 0
        if Hmax (ind) == H(row,col)
            found = 1;
        elseif col<size(H,2)
            col = col+1;
        else
            row = row + 1; col = 1;
        end
    end
    X(row, col) = 1;
end

end


% function Xx = hist_counter(X, H, threshold)
% for i = 1:size(H,1)
%     for j = 1:size(H,2)
%         if H(i,j) < 1
%             H(i,j) = 0;
%         end
%     end
% end
% 
% Xmax = sort(reshape(X,1,[]),'descend');
% Hmax = zeros(size(Xmax));
% 
% for ind = 1:length(Xmax)
%     found = 0; row = 1; col = 1;
%     while found == 0
%         if Xmax (ind) == X(row,col)
%             found = 1;
%         elseif col<size(H,2)
%             col = col+1;
%         else
%             row = row + 1; col = 1;
%         end
%     end
%     Hmax(ind) = H(row, col);
% end
% 
% Hmax_cum = cumsum(Hmax);
% 
% i = 1;
% while Hmax_cum(i) < threshold*Hmax_cum(numel(Hmax_cum))
%     i = i+1;
% end
% 
% Xx = zeros(size(H));
% for ind = 1:i
%     found = 0; row = 1; col = 1;
%     while found == 0
%         if Hmax (ind) == H(row,col)
%             found = 1;
%         elseif col<size(H,2)
%             col = col+1;
%         else
%             row = row + 1; col = 1;
%         end
%     end
%     Xx(row, col) = 1;
% end
% 
% end