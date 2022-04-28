clear all; close all;

n = 50;
dt = 0.005;
dx = 1/n;
v1 = zeros(n+1,n+1);
v2 = zeros(n+1,n+1);
nu = 0; %-4e-4;

kx = pi;
ky = pi;

x0 = 0.2; y0 = 0.4;
sigma = 100.0;
u = zeros(n+1,n+1);
%u(5,5) = 100;
for i = 1:n+1
    for j = 1:n+1
        v1(i,j) = -ky*sin(kx*(i-1)*dx)*cos(ky*(j-1)*dx);
        v2(i,j) = kx*cos(kx*(i-1)*dx)*sin(ky*(j-1)*dx);
        u(i,j) = exp(-sigma*(((i-1)*dx-x0)*((i-1)*dx-x0) + ((j-1)*dx-y0)*((j-1)*dx-y0)));
    end
end

for i = 1:100
    rhs = compute_rhs(u, n, v1, v2, dt, nu, dx);
    resnorm = 1;
    while resnorm > 1e-5
        u = gauss_seidel(u, rhs, n, v1, v2, dt, nu, dx);
        res = residual(u, rhs, n, v1, v2, dt, nu, dx);
        resnorm = compute_norm(res, n);
    end
    pcolor(u)
    pause(0.05)
end



function rr = r(h,k)
    rr = 1.0/2*k/(h*h);
end


function aa = a(v, nu, h, r)
    aa = r*(-v*h/2.0+nu);
end

function bb = b(v, nu, h, r)
    bb = r*(v*h/2.0+nu);
end


% compute rhs with u
function rhs = compute_rhs(u, n, v1, v2, k, nu, h)
    % k: time step (dt)
    % h: spatial discretization step (dx=dy)
    % r: dt/(2*dx*dx)
    rr = r(h,k);
    rhs = zeros(n+1,n+1);
    for i=2:n
        for j=2:n
            aa = a(v2(i,j),nu,h,rr);
            bb = b(v2(i,j),nu,h,rr);
            cc = a(v1(i,j),nu,h,rr);
            dd = b(v1(i,j),nu,h,rr);
            rhs(i,j) = (1.0+4.0*rr*nu)*u(i,j) - cc*u(i-1,j) - aa*u(i,j-1) - dd*u(i+1,j) - bb*u(i,j+1);
        end
    end
end

function res = residual(u, rhs, n, v1, v2, k, nu, h)
    % k: time step (dt)
    % h: spatial discretization step (dx=dy)
    % r: dt/(2*dx*dx)
    rr = r(h,k);
    res = zeros(n+1,n+1);
    for i = 2:n
        for j = 2:n 
            aa = a(v2(i,j),nu,h,rr);
            bb = b(v2(i,j),nu,h,rr);
            cc = a(v1(i,j),nu,h,rr);
            dd = b(v1(i,j),nu,h,rr);
            res(i,j) = rhs(i,j) - ((1.0-4.0*rr*nu)*u(i,j) + cc*u(i-1,j) + aa*u(i,j-1) + dd*u(i+1,j) + bb*u(i,j+1));
        end
    end
end


function resnorm = compute_norm(res, n)
    tmp = 0.0;
    for i = 2:n
        for j = 2:n
            tmp = tmp + res(i,j)*res(i,j);
        end
    end
    resnorm = sqrt(tmp);
end

function u = gauss_seidel(u, rhs, n, v1, v2, k, nu, h)
    rr = r(h,k);
    % UPDATING RED POINTS
    % here we're assuming top left is red
    % interior
    for i = 2:n
        j = 2-mod(i,2)+1;
    	while j < n+1
            aa = a(v2(i,j),nu,h,rr);
            bb = b(v2(i,j),nu,h,rr);
            cc = a(v1(i,j),nu,h,rr);
            dd = b(v1(i,j),nu,h,rr);
            u(i,j) = (rhs(i,j) - cc*u(i-1,j) - aa*u(i,j-1) - dd*u(i+1,j) - bb*u(i,j+1))/(1.0-4.0*rr*nu);
            j = j+2;
        end
    end

    % UPDATING BLACK POINTS
    % interior
    for i = 2:n
        j = 1+mod(i,2)+1;
    	while j < n+1
            aa = a(v2(i,j),nu,h,rr);
            bb = b(v2(i,j),nu,h,rr);
            cc = a(v1(i,j),nu,h,rr);
            dd = b(v1(i,j),nu,h,rr);
            u(i,j) = (rhs(i,j) - cc*u(i-1,j) - aa*u(i,j-1) - dd*u(i+1,j) - bb*u(i,j+1))/(1.0-4.0*rr*nu);
            j = j+2;
        end
    end
end
